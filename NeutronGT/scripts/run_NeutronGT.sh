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
    echo "Usage: bash $0 <devices> [--arxiv|--amazon|--reddit|--products|--papers100M ...] [--GT|--GPH_Slim|--GPH_Large|--ALL] [--preprocess_only] [--refresh_preprocess_cache]"
    echo "Example: bash $0 0,1,2,3"
    echo "         bash $0 0,1,2,3 --arxiv --products --GPH_Slim"
    echo "         bash $0 0,1,2,3 --papers100M --ALL --refresh_preprocess_cache"
    exit 1
fi
shift

DATASET_DIR=./dataset/
LOG_DIR=NeutronGT_logs/run_NeutronGT
RUN_TAG=$(date +%Y%m%d_%H%M)
PREPROCESS_ONLY=0
REFRESH_PREPROCESS_CACHE=0
SELECTED_DATASET_FLAGS=()
MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arxiv|--amazon|--reddit|--products|--papers100M) SELECTED_DATASET_FLAGS+=("$1") ;;
        --GT|--GPH_Slim|--GPH_Large) MODELS+=("${1:2}") ;;
        --ALL) MODELS=(GT GPH_Slim GPH_Large) ;;
        --preprocess_only) PREPROCESS_ONLY=1 ;;
        --refresh_preprocess_cache) REFRESH_PREPROCESS_CACHE=1 ;;
        *)
            echo "Usage: bash $0 <devices> [--arxiv|--amazon|--reddit|--products|--papers100M ...] [--GT|--GPH_Slim|--GPH_Large|--ALL] [--preprocess_only] [--refresh_preprocess_cache]" >&2
            echo "Error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ ${#SELECTED_DATASET_FLAGS[@]} -eq 0 ]; then
    DATASET_FLAGS=(--arxiv --amazon --reddit --products --papers100M)
else
    DATASET_FLAGS=("${SELECTED_DATASET_FLAGS[@]}")
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(GT GPH_Slim GPH_Large)
fi

IFS=, read -r -a GPU_LIST <<< "$DEVICES"
GPU_NUM=${#GPU_LIST[@]}

mkdir -p "${LOG_DIR}"

WINDOW_AUG_STRATEGY="ours"
USE_CACHE=1
USE_PREPROCESS_CACHE=1
ATTN_TYPE="sparse"
PPR_BACKEND="appnp"
PPR_TOPK=5
PPR_ALPHA=0.85
PPR_NUM_ITER=10

resolve_dataset() {
    case "$1" in
        --arxiv) echo "ogbn-arxiv" ;;
        --amazon) echo "AmazonProducts" ;;
        --reddit) echo "reddit" ;;
        --products) echo "ogbn-products" ;;
        --papers100M) echo "ogbn-papers100M" ;;
        *) return 1 ;;
    esac
}

resolve_model_params() {
    case "$1" in
        GT) echo "gt_sw 4 128 128 8" ;;
        GPH_Slim) echo "graphormer 4 64 64 8" ;;
        GPH_Large) echo "graphormer 12 768 768 32" ;;
        *) return 1 ;;
    esac
}

resolve_run_params() {
    local dataset="$1"
    local model_alias="$2"

    if [ "$dataset" = "ogbn-papers100M" ]; then
        if [ "$model_alias" = "GPH_Large" ]; then
            echo "8192 40 2048 64 640 0.10 0.05 0.05"
        else
            echo "2048 40 2048 64 640 0.10 0.05 0.05"
        fi
        return 0
    fi

    if [ "$dataset" = "AmazonProducts" ]; then
        if [ "$model_alias" = "GPH_Large" ]; then
            echo "1024 40 8192 5 120 0.15 0.075 0.075"
        else
            echo "512 40 8192 5 120 0.15 0.075 0.075"
        fi
    elif [ "$dataset" = "ogbn-arxiv" ]; then
        if [ "$model_alias" = "GPH_Large" ]; then
            echo "32 40 8192 5 120 0.30 0.15 0.15"
        else
            echo "16 40 8192 5 120 0.30 0.15 0.15"
        fi
    elif [ "$dataset" = "ogbn-products" ]; then
        if [ "$model_alias" = "GPH_Large" ]; then
            echo "2048 40 8192 5 120 0.15 0.075 0.075"
        else
            echo "1024 40 8192 5 120 0.15 0.075 0.075"
        fi
    elif [ "$dataset" = "reddit" ]; then
        if [ "$model_alias" = "GPH_Large" ]; then
            echo "128 40 8192 5 120 0.20 0.10 0.10"
        else
            echo "64 40 8192 5 120 0.20 0.10 0.10"
        fi
    else
        return 1
    fi
}

for DATASET_FLAG in "${DATASET_FLAGS[@]}"; do
    DATASET=$(resolve_dataset "${DATASET_FLAG}")

    for MODEL_ALIAS in "${MODELS[@]}"; do
        read -r MODEL N_LAYERS HIDDEN_DIM FFN_DIM NUM_HEADS <<< "$(resolve_model_params "${MODEL_ALIAS}")"
        read -r NPARTS EPOCHS PPR_BATCH_SIZE PPR_ITER_TOPK TIMEOUT WINDOW_EXTRA_RATIO WINDOW_RELATED_RATIO WINDOW_HUB_RATIO <<< "$(resolve_run_params "${DATASET}" "${MODEL_ALIAS}")"

        MODE_LABEL="train"
        if [ "$PREPROCESS_ONLY" -eq 1 ]; then
            MODE_LABEL="preprocess"
        fi
        if [ "$REFRESH_PREPROCESS_CACHE" -eq 1 ]; then
            MODE_LABEL="${MODE_LABEL}_refresh"
        fi

        LOG_FILE="${LOG_DIR}/${DATASET}_${MODEL_ALIAS}_${WINDOW_AUG_STRATEGY}_e${EPOCHS}_nparts${NPARTS}_${MODE_LABEL}_${RUN_TAG}.log"
        MASTER_PORT=$((8000 + RANDOM % 1000))

        echo "============================================================="
        echo "NeutronGT"
        echo "Dataset: ${DATASET}"
        echo "Model: ${MODEL_ALIAS}"
        echo "layers=${N_LAYERS} hidden=${HIDDEN_DIM} ffn=${FFN_DIM} heads=${NUM_HEADS}"
        echo "epochs=${EPOCHS} n_parts=${NPARTS}"
        echo "window_aug=${WINDOW_AUG_STRATEGY} extra=${WINDOW_EXTRA_RATIO} related=${WINDOW_RELATED_RATIO} hub=${WINDOW_HUB_RATIO}"
        echo "cache=${USE_CACHE} preprocess_cache=${USE_PREPROCESS_CACHE} refresh=${REFRESH_PREPROCESS_CACHE}"
        echo "ppr_backend=${PPR_BACKEND} ppr_topk=${PPR_TOPK} ppr_batch=${PPR_BATCH_SIZE} ppr_iter_topk=${PPR_ITER_TOPK}"
        echo "GPUs=${GPU_NUM} CUDA_VISIBLE_DEVICES=${DEVICES} timeout=${TIMEOUT}m"
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
            --window_aug_strategy "${WINDOW_AUG_STRATEGY}" \
            --window_extra_node_ratio "${WINDOW_EXTRA_RATIO}" \
            --window_related_ratio "${WINDOW_RELATED_RATIO}" \
            --window_hub_ratio "${WINDOW_HUB_RATIO}" \
            --ppr_backend "${PPR_BACKEND}" \
            --ppr_topk "${PPR_TOPK}" \
            --ppr_alpha "${PPR_ALPHA}" \
            --ppr_num_iterations "${PPR_NUM_ITER}" \
            --ppr_batch_size "${PPR_BATCH_SIZE}" \
            --ppr_iter_topk "${PPR_ITER_TOPK}" \
            --preprocess_only "${PREPROCESS_ONLY}" \
            --distributed-backend nccl \
            --distributed-timeout-minutes "${TIMEOUT}" \
            > "${LOG_FILE}" 2>&1

        EXIT_CODE=$?
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "[${DATASET} ${MODEL_ALIAS}] Failed (exit ${EXIT_CODE}), stopping. Check ${LOG_FILE}"
            exit ${EXIT_CODE}
        fi
        echo "[${DATASET} ${MODEL_ALIAS}] Done."
    done
done

echo "All NeutronGT runs done."
