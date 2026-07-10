#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")/.."

export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# ==================== Usage ====================
#   bash scripts/run_papers100M.sh 0,1,2,3
# ===============================================

DEVICES=${1-}
if [ -z "$DEVICES" ] || [[ "$DEVICES" == -* ]]; then
    echo "Usage: bash $0 <devices>"
    echo "Example: bash $0 0,1,2,3"
    exit 1
fi

IFS=, read -r -a GPU_LIST <<< "$DEVICES"
GPU_NUM=${#GPU_LIST[@]}

DATASET="ogbn-papers100M"
DATASET_DIR=./dataset/
LOG_DIR=NeutronGT_logs
RUN_TAG=$(date +%Y%m%d_%H%M)
EPOCHS=21
PREPROCESS_ONLY=0

# 三个模型共享的 PPR 参数
ATTN_TYPE="sparse"
PPR_BACKEND="appnp"
PPR_TOPK=5
PPR_ALPHA=0.85
PPR_NUM_ITER=10
PPR_BATCH_SIZE=2048
PPR_ITER_TOPK=64
USE_CACHE=1
TIMEOUT=640

mkdir -p "${LOG_DIR}"

for MODEL_ALIAS in GT GPH_Slim GPH_Large; do
    case "$MODEL_ALIAS" in
        "GT")
            MODEL="gt_sw"
            N_LAYERS=4; HIDDEN_DIM=128; FFN_DIM=128; NUM_HEADS=8
            NPARTS=4096; RELATED_TOPK=4
            ;;
        "GPH_Slim")
            MODEL="graphormer"
            N_LAYERS=4; HIDDEN_DIM=64; FFN_DIM=64; NUM_HEADS=8
            NPARTS=4096; RELATED_TOPK=4
            ;;
        "GPH_Large")
            MODEL="graphormer"
            N_LAYERS=12; HIDDEN_DIM=768; FFN_DIM=768; NUM_HEADS=32
            NPARTS=8192; RELATED_TOPK=2
            ;;
    esac

    LOG_FILE="${LOG_DIR}/${DATASET}_${MODEL_ALIAS}_e${EPOCHS}_${RUN_TAG}.log"
    MASTER_PORT=$((8000 + RANDOM % 1000))

    echo "============================================================="
    echo "  NeutronGT - papers100M  ${MODEL_ALIAS}"
    echo "============================================================="
    echo "  layers=${N_LAYERS} hidden=${HIDDEN_DIM} ffn=${FFN_DIM} heads=${NUM_HEADS}"
    echo "  n_parts=${NPARTS} related_topk=${RELATED_TOPK} epochs=${EPOCHS}"
    echo "  GPUs=${GPU_NUM}  timeout=${TIMEOUT}m  log=${LOG_FILE}"
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
        --use_preprocess_cache 0 \
        --n_parts "${NPARTS}" \
        --related_nodes_topk_rate "${RELATED_TOPK}" \
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
        echo "[${MODEL_ALIAS}] Failed (exit ${EXIT_CODE}), stopping. Check ${LOG_FILE}"
        exit ${EXIT_CODE}
    fi
    echo "[${MODEL_ALIAS}] Done."
done

echo "All three models finished."
