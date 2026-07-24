#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")/.."

export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

DEVICES=${1-}
if [ -z "$DEVICES" ] || [[ "$DEVICES" == -* ]]; then
    echo "Usage: bash $0 <devices> --GT|--GPH_Slim|--GPH_Large [--arxiv|--amazon|--reddit|--products ...] [--refresh_preprocess_cache] [--preprocess_only]"
    echo "Example: bash $0 0,1,2,3 --GPH_Slim"
    echo "         bash $0 0,1,2,3 --GPH_Slim --arxiv --products"
    echo "         bash $0 0,1,2,3 --GPH_Large --reddit --refresh_preprocess_cache"
    exit 1
fi
shift

MODEL_INPUT=""
REFRESH_PREPROCESS_CACHE=0
PREPROCESS_ONLY=0
SELECTED_DATASET_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --GT) MODEL_INPUT="GT" ;;
        --GPH_Slim) MODEL_INPUT="GPH_Slim" ;;
        --GPH_Large) MODEL_INPUT="GPH_Large" ;;
        --arxiv|--amazon|--reddit|--products) SELECTED_DATASET_FLAGS+=("$1") ;;
        --refresh_preprocess_cache) REFRESH_PREPROCESS_CACHE=1 ;;
        --preprocess_only) PREPROCESS_ONLY=1 ;;
        *)
            echo "Usage: bash $0 <devices> --GT|--GPH_Slim|--GPH_Large [--arxiv|--amazon|--reddit|--products ...] [--refresh_preprocess_cache] [--preprocess_only]" >&2
            echo "Error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$MODEL_INPUT" ]; then
    echo "Error: model is required." >&2
    echo "Usage: bash $0 <devices> --GT|--GPH_Slim|--GPH_Large [--arxiv|--amazon|--reddit|--products ...] [--refresh_preprocess_cache] [--preprocess_only]" >&2
    exit 1
fi

case "$MODEL_INPUT" in
    "GT")
        MODEL_ALIAS="GT"
        MODEL="gt_sw"
        N_LAYERS=4
        HIDDEN_DIM=128
        FFN_DIM=128
        NUM_HEADS=8
        ;;
    "GPH_Slim")
        MODEL_ALIAS="GPH_Slim"
        MODEL="graphormer"
        N_LAYERS=4
        HIDDEN_DIM=64
        FFN_DIM=64
        NUM_HEADS=8
        ;;
    "GPH_Large")
        MODEL_ALIAS="GPH_Large"
        MODEL="graphormer"
        N_LAYERS=12
        HIDDEN_DIM=768
        FFN_DIM=768
        NUM_HEADS=32
        ;;
    *)
        echo "Error: unsupported model: ${MODEL_INPUT}" >&2
        exit 1
        ;;
esac

DATASET_DIR=./dataset/
LOG_DIR=NeutronGT_logs/window_aug_comparison
RUN_TAG=$(date +%Y%m%d_%H%M)
EPOCHS=500
ATTN_TYPE="sparse"
USE_CACHE=1
USE_PREPROCESS_CACHE=0
TIMEOUT=120
PPR_BATCH_SIZE=8192
PPR_ITER_TOPK=5

if [ ${#SELECTED_DATASET_FLAGS[@]} -eq 0 ]; then
    DATASET_FLAGS=(--arxiv --amazon --reddit --products)
else
    DATASET_FLAGS=("${SELECTED_DATASET_FLAGS[@]}")
fi
STRATEGIES=(ours hub random related)

IFS=, read -r -a GPU_LIST <<< "$DEVICES"
GPU_NUM=${#GPU_LIST[@]}

mkdir -p "${LOG_DIR}"

resolve_dataset() {
    case "$1" in
        --arxiv) echo "ogbn-arxiv" ;;
        --amazon) echo "AmazonProducts" ;;
        --reddit) echo "reddit" ;;
        --products) echo "ogbn-products" ;;
        *) return 1 ;;
    esac
}

resolve_window_params() {
    local dataset="$1"
    local model_alias="$2"
    if [ "$dataset" = "AmazonProducts" ]; then
        if [ "$model_alias" = "GPH_Large" ]; then
            echo "800 0.10 0.05 0.05"
        else
            echo "256 0.10 0.05 0.05"
        fi
    elif [ "$dataset" = "ogbn-arxiv" ]; then
        if [ "$model_alias" = "GPH_Large" ]; then
            echo "32 0.30 0.15 0.15"
        else
            echo "16 0.30 0.15 0.15"
        fi
    elif [ "$dataset" = "ogbn-products" ]; then
        if [ "$model_alias" = "GPH_Large" ]; then
            echo "1024 0.10 0.05 0.05"
        else
            echo "512 0.10 0.05 0.05"
        fi
    elif [ "$dataset" = "reddit" ]; then
        if [ "$model_alias" = "GPH_Large" ]; then
            echo "96 0.15 0.075 0.075"
        else
            echo "48 0.15 0.075 0.075"
        fi
    else
        return 1
    fi
}

for DATASET_FLAG in "${DATASET_FLAGS[@]}"; do
    DATASET=$(resolve_dataset "${DATASET_FLAG}")
    read -r NPARTS WINDOW_EXTRA_RATIO WINDOW_RELATED_RATIO WINDOW_HUB_RATIO <<< "$(resolve_window_params "${DATASET}" "${MODEL_ALIAS}")"

    for STRATEGY in "${STRATEGIES[@]}"; do
        MODE_LABEL="train"
        if [ "$PREPROCESS_ONLY" -eq 1 ]; then
            MODE_LABEL="preprocess"
        fi
        if [ "$REFRESH_PREPROCESS_CACHE" -eq 1 ]; then
            MODE_LABEL="${MODE_LABEL}_refresh"
        fi

        LOG_FILE="${LOG_DIR}/${DATASET}_${MODEL_ALIAS}_${STRATEGY}_e${EPOCHS}_nparts${NPARTS}_${MODE_LABEL}_${RUN_TAG}.log"
        MASTER_PORT=$((8000 + RANDOM % 1000))

        echo "============================================================="
        echo "Window augmentation comparison"
        echo "Dataset: ${DATASET}"
        echo "Model: ${MODEL_ALIAS}"
        echo "Strategy: ${STRATEGY}"
        echo "n_parts=${NPARTS} epochs=${EPOCHS}"
        echo "window extra=${WINDOW_EXTRA_RATIO} related=${WINDOW_RELATED_RATIO} hub=${WINDOW_HUB_RATIO}"
        echo "cache=${USE_CACHE} preprocess_cache=${USE_PREPROCESS_CACHE} refresh=${REFRESH_PREPROCESS_CACHE}"
        echo "GPUs=${GPU_NUM} CUDA_VISIBLE_DEVICES=${DEVICES}"
        echo "Log: ${LOG_FILE}"
        echo "============================================================="

        CUDA_VISIBLE_DEVICES="${DEVICES}" torchrun \
            --nproc_per_node="${GPU_NUM}" \
            --master_port="${MASTER_PORT}" \
            main_sp_node_level_ppr.py \
            --dataset "${DATASET}" \
            --dataset_dir "${DATASET_DIR}" \
            --model "${MODEL}" \
            --attn_type "${ATTN_TYPE}" \
            --n_layers "${N_LAYERS}" \
            --hidden_dim "${HIDDEN_DIM}" \
            --ffn_dim "${FFN_DIM}" \
            --num_heads "${NUM_HEADS}" \
            --epochs "${EPOCHS}" \
            --use_cache "${USE_CACHE}" \
            --use_preprocess_cache "${USE_PREPROCESS_CACHE}" \
            --refresh_preprocess_cache "${REFRESH_PREPROCESS_CACHE}" \
            --n_parts "${NPARTS}" \
            --window_aug_strategy "${STRATEGY}" \
            --window_extra_node_ratio "${WINDOW_EXTRA_RATIO}" \
            --window_related_ratio "${WINDOW_RELATED_RATIO}" \
            --window_hub_ratio "${WINDOW_HUB_RATIO}" \
            --ppr_backend appnp \
            --ppr_topk 5 \
            --ppr_alpha 0.85 \
            --ppr_num_iterations 10 \
            --ppr_batch_size "${PPR_BATCH_SIZE}" \
            --ppr_iter_topk "${PPR_ITER_TOPK}" \
            --preprocess_only "${PREPROCESS_ONLY}" \
            --distributed-backend nccl \
            --distributed-timeout-minutes "${TIMEOUT}" \
            > "${LOG_FILE}" 2>&1

        EXIT_CODE=$?
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "[${DATASET} ${MODEL_ALIAS} ${STRATEGY}] Failed (exit ${EXIT_CODE}), stopping. Check ${LOG_FILE}"
            exit ${EXIT_CODE}
        fi
        echo "[${DATASET} ${MODEL_ALIAS} ${STRATEGY}] Done."
    done
done

echo "All window augmentation comparisons done."
