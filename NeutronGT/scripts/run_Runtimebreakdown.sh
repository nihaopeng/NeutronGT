#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")/.."

export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# 1. Parse the positional argument for GPUs
DEVICES=${1-}
if [ -z "$DEVICES" ] || [[ "$DEVICES" == -* ]]; then
    echo "Error: CUDA_VISIBLE_DEVICES argument is required."
    echo "Usage: bash $0 <devices> [dataset] [model]"
    echo "Datasets: --arxiv | --amazon | --reddit | --products"
    echo "Models:   --GPH_Large"
    echo "Example:  bash $0 0,1,2,3 --arxiv --GPH_Large"
    exit 1
fi
shift

DATASET_INPUT=""
MODEL_INPUT=""
DATASET_DIR=./dataset/
LOG_DIR=NeutronGT_logs
RUN_TAG=$(date +%Y%m%d_%H%M)

# 2. Parse long flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arxiv)    DATASET_INPUT="ogbn-arxiv" ;;
        --amazon)   DATASET_INPUT="AmazonProducts" ;;
        --reddit)   DATASET_INPUT="reddit" ;;
        --products) DATASET_INPUT="ogbn-products" ;;
        --GPH_Large) MODEL_INPUT="GPH_Large" ;;
        *)
            echo "Usage: bash $0 <devices> [dataset] [model]"
            echo "Datasets: --arxiv | --amazon | --reddit | --products"
            echo "Models:   --GPH_Large"
            echo "Example:  bash $0 0,1,2,3 --arxiv --GPH_Large"
            echo "Error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

# 3. Validate required selections
if [ -z "$DATASET_INPUT" ]; then
    echo "Error: dataset is required." >&2
    echo "Usage: bash $0 <devices> --arxiv|--amazon|--reddit|--products --GPH_Large" >&2
    exit 1
fi

if [ -z "$MODEL_INPUT" ]; then
    echo "Error: model is required." >&2
    echo "Usage: bash $0 <devices> --arxiv|--amazon|--reddit|--products --GPH_Large" >&2
    exit 1
fi

# ================= Parameter Mapping =================

dataset="$DATASET_INPUT"
MODEL_ALIAS="GPH_Large"
MODEL="graphormer"
N_LAYERS=12
HIDDEN_DIM=768
FFN_DIM=768
NUM_HEADS=32
ATTN_TYPE="sparse"
USE_CACHE=1
EPOCHS=500

if [ "$dataset" = "AmazonProducts" ]; then
    NPARTS=400
    RELATED_TOPK=4
elif [ "$dataset" = "ogbn-arxiv" ]; then
    NPARTS=32
    RELATED_TOPK=8
elif [ "$dataset" = "ogbn-products" ]; then
    NPARTS=512
    RELATED_TOPK=4
elif [ "$dataset" = "reddit" ]; then
    NPARTS=80
    RELATED_TOPK=4
else
    echo "Error: unsupported dataset: $dataset" >&2
    exit 1
fi

# ================= Main Execution =================

IFS=, read -r -a GPU_LIST <<< "$DEVICES"
GPU_NUM=${#GPU_LIST[@]}

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${dataset}_${MODEL_ALIAS}_Runtimebreakdown_sparse_cache_500ep_nparts${NPARTS}_rtopk${RELATED_TOPK}_${RUN_TAG}.log"
MASTER_PORT=$((8000 + RANDOM % 1000))

echo "-------------------------------------------------------------"
echo "Dataset: ${dataset}"
echo "Model: ${MODEL_ALIAS}"
echo "Attention: ${ATTN_TYPE}"
echo "Cache: ${USE_CACHE}"
echo "Epochs: ${EPOCHS}"
echo "GPUs: ${GPU_NUM} (CUDA_VISIBLE_DEVICES=${DEVICES})"
echo "Log: ${LOG_FILE}"
echo "-------------------------------------------------------------"

CUDA_VISIBLE_DEVICES="${DEVICES}" torchrun   --nproc_per_node="${GPU_NUM}"   --master_port="${MASTER_PORT}"   main_sp_node_level_ppr.py   --dataset "${dataset}"   --dataset_dir "${DATASET_DIR}"   --model "${MODEL}"   --attn_type "${ATTN_TYPE}"   --n_layers "${N_LAYERS}"   --hidden_dim "${HIDDEN_DIM}"   --ffn_dim "${FFN_DIM}"   --num_heads "${NUM_HEADS}"   --epochs "${EPOCHS}"   --use_cache "${USE_CACHE}"   --use_preprocess_cache 0   --n_parts "${NPARTS}"   --related_nodes_topk_rate "${RELATED_TOPK}"   --ppr_backend appnp   --ppr_topk 5   --ppr_alpha 0.85   --ppr_num_iterations 10   --ppr_batch_size 8192   --ppr_iter_topk 5   --distributed-backend nccl   --distributed-timeout-minutes 120   > "${LOG_FILE}" 2>&1

if [ $? -eq 0 ]; then
    echo "Status: Success"
else
    echo "Status: Failed. Check ${LOG_FILE}"
fi
