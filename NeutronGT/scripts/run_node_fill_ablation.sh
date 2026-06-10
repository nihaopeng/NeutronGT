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
    echo "Error: CUDA_VISIBLE_DEVICES argument is required."
    echo "Usage: bash $0 <devices> [dataset] [model] [mode]"
    echo "Datasets: --arxiv | --products"
    echo "Models:   --GT | --GPH_Slim | --GPH_Large"
    echo "Modes:    --random | --none | --highdeg"
    exit 1
fi
shift

DATASET_INPUT=""
MODEL_INPUT=""
MODE_INPUT=""
DATASET_DIR=./dataset/
LOG_DIR=NeutronGT_logs/node_fill_ablation
RUN_TAG=$(date +%Y%m%d_%H%M)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arxiv)    DATASET_INPUT="ogbn-arxiv" ;;
        --products) DATASET_INPUT="ogbn-products" ;;
        --GT)       MODEL_INPUT="GT" ;;
        --GPH_Slim) MODEL_INPUT="GPH_Slim" ;;
        --GPH_Large) MODEL_INPUT="GPH_Large" ;;
        --random)   MODE_INPUT="random" ;;
        --none)     MODE_INPUT="none" ;;
        --highdeg)  MODE_INPUT="highdeg" ;;
        *)
            echo "Usage: bash $0 <devices> [dataset] [model] [mode]"
            echo "Datasets: --arxiv | --products"
            echo "Models:   --GT | --GPH_Slim | --GPH_Large"
            echo "Modes:    --random | --none | --highdeg"
            echo "Error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$DATASET_INPUT" ] || [ -z "$MODEL_INPUT" ] || [ -z "$MODE_INPUT" ]; then
    echo "Error: dataset, model and mode are required." >&2
    exit 1
fi

dataset="$DATASET_INPUT"

case "$MODEL_INPUT" in
    "GT")
        MODEL_ALIAS="GT"
        MODEL="gt_sw"
        N_LAYERS=4
        HIDDEN_DIM=128
        FFN_DIM=128
        NUM_HEADS=8
        ATTN_TYPE="sparse"
        EPOCHS=500
        ;;
    "GPH_Slim")
        MODEL_ALIAS="GPH_Slim"
        MODEL="graphormer"
        N_LAYERS=4
        HIDDEN_DIM=64
        FFN_DIM=64
        NUM_HEADS=8
        ATTN_TYPE="sparse"
        EPOCHS=500
        ;;
    "GPH_Large")
        MODEL_ALIAS="GPH_Large"
        MODEL="graphormer"
        N_LAYERS=12
        HIDDEN_DIM=768
        FFN_DIM=768
        NUM_HEADS=32
        ATTN_TYPE="sparse"
        EPOCHS=200
        ;;
    *)
        echo "Error: unsupported model: $MODEL_INPUT" >&2
        exit 1
        ;;
esac

if [ "$dataset" = "ogbn-arxiv" ]; then
    if [ "$MODEL_ALIAS" = "GT" ]; then
        NPARTS=16
        RELATED_TOPK=8
    elif [ "$MODEL_ALIAS" = "GPH_Slim" ]; then
        NPARTS=16
        RELATED_TOPK=8
    elif [ "$MODEL_ALIAS" = "GPH_Large" ]; then
        NPARTS=32
        RELATED_TOPK=8
    fi
elif [ "$dataset" = "ogbn-products" ]; then
    if [ "$MODEL_ALIAS" = "GT" ]; then
        NPARTS=128
        RELATED_TOPK=6
    elif [ "$MODEL_ALIAS" = "GPH_Slim" ]; then
        NPARTS=128
        RELATED_TOPK=6
    elif [ "$MODEL_ALIAS" = "GPH_Large" ]; then
        NPARTS=512
        RELATED_TOPK=4
    fi
else
    echo "Error: unsupported dataset: $dataset" >&2
    exit 1
fi

MODE_FLAG=""
MODE_TAG=""
case "$MODE_INPUT" in
    "random")
        MODE_FLAG="--random_replace_window_nodes 1"
        MODE_TAG="random"
        ;;
    "none")
        MODE_FLAG="--disable_window_node_expansion 1"
        MODE_TAG="none"
        ;;
    "highdeg")
        MODE_FLAG="--high_degree_replace_window_nodes 1"
        MODE_TAG="highdeg"
        ;;
    *)
        echo "Error: unsupported mode: $MODE_INPUT" >&2
        exit 1
        ;;
esac

IFS=, read -r -a GPU_LIST <<< "$DEVICES"
GPU_NUM=${#GPU_LIST[@]}

EXP_NAME="${dataset}_${MODEL_ALIAS}_${MODE_TAG}_nparts${NPARTS}_rtopk${RELATED_TOPK}"
MODEL_DIR="./model_ckpt/${EXP_NAME}"
CHECKPOINT_DIR="./model_ckpt/${EXP_NAME}/resume_ckpt"
mkdir -p "${LOG_DIR}" "${MODEL_DIR}" "${CHECKPOINT_DIR}"

LOG_FILE="${LOG_DIR}/${EXP_NAME}_${RUN_TAG}.log"
MASTER_PORT=$((8000 + RANDOM % 1000))

echo "-------------------------------------------------------------"
echo "Dataset: ${dataset}"
echo "Model: ${MODEL_ALIAS}"
echo "Mode: ${MODE_TAG}"
echo "GPUs: ${GPU_NUM} (CUDA_VISIBLE_DEVICES=${DEVICES})"
echo "Experiment: ${EXP_NAME}"
echo "Model dir: ${MODEL_DIR}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Log: ${LOG_FILE}"
echo "-------------------------------------------------------------"

CUDA_VISIBLE_DEVICES="${DEVICES}" torchrun \
  --nproc_per_node="${GPU_NUM}" \
  --master_port="${MASTER_PORT}" \
  main_sp_node_level_ppr.py \
  --dataset "${dataset}" \
  --dataset_dir "${DATASET_DIR}" \
  --model "${MODEL}" \
  --attn_type "${ATTN_TYPE}" \
  --n_layers "${N_LAYERS}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --ffn_dim "${FFN_DIM}" \
  --num_heads "${NUM_HEADS}" \
  --epochs "${EPOCHS}" \
  --use_cache 1 \
  --use_preprocess_cache 1 \
  --n_parts "${NPARTS}" \
  --related_nodes_topk_rate "${RELATED_TOPK}" \
  --ppr_backend appnp \
  --ppr_topk 5 \
  --ppr_alpha 0.85 \
  --ppr_num_iterations 10 \
  --ppr_batch_size 8192 \
  --ppr_iter_topk 5 \
  --distributed-backend nccl \
  --distributed-timeout-minutes 120 \
  --save_checkpoint \
  --resume_latest \
  --save_latest_only \
  --model_dir "${MODEL_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  ${MODE_FLAG} \
  > "${LOG_FILE}" 2>&1

if [ $? -eq 0 ]; then
    echo "Status: Success"
else
    echo "Status: Failed. Check ${LOG_FILE}"
fi
