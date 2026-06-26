#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")/.."

export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# ==================== Usage ====================
# bash scripts/run_papers100M.sh <devices> [--GT|--GPH_Slim|--GPH_Large] [--preprocess_only]
#
# 阶段 1（推荐先测试）：仅预处理，不训练
#   bash scripts/run_papers100M.sh 0,1,2,3 --GPH_Slim --preprocess_only
#
# 阶段 2：完整训练
#   bash scripts/run_papers100M.sh 0,1,2,3 --GPH_Slim
#
# n_parts 选择依据（参照 ogbn-products: 2.4M/128≈19K nodes/window）:
#   GPH_Slim:  n_parts=4096 → ~31K nodes/window → ~1.1 GB GPU
#   GT:        n_parts=4096 → ~31K nodes/window → ~2.0 GB GPU
#   GPH_Large: n_parts=8192 → ~16K nodes/window → ~9.5 GB GPU
# ===============================================

DEVICES=${1-}
if [ -z "$DEVICES" ] || [[ "$DEVICES" == -* ]]; then
    echo "Error: CUDA_VISIBLE_DEVICES argument is required."
    echo "Usage: bash $0 <devices> [--GT|--GPH_Slim|--GPH_Large] [--preprocess_only]"
    echo "Example: bash $0 0,1,2,3 --GPH_Slim"
    echo "         bash $0 0,1,2,3 --GT --preprocess_only"
    exit 1
fi
shift

# ==================== Argument Parsing ====================

MODEL_INPUT=""
PREPROCESS_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --GT)           MODEL_INPUT="GT" ;;
        --GPH_Slim)     MODEL_INPUT="GPH_Slim" ;;
        --GPH_Large)    MODEL_INPUT="GPH_Large" ;;
        --preprocess_only) PREPROCESS_ONLY=1 ;;
        *)
            echo "Error: unknown argument: $1" >&2
            echo "Usage: bash $0 <devices> [--GT|--GPH_Slim|--GPH_Large] [--preprocess_only]" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$MODEL_INPUT" ]; then
    echo "Error: model is required." >&2
    echo "Usage: bash $0 <devices> --GT|--GPH_Slim|--GPH_Large [--preprocess_only]" >&2
    exit 1
fi

# ==================== Parameter Mapping ====================

DATASET="ogbn-papers100M"
DATASET_DIR=./dataset/
LOG_DIR=NeutronGT_logs
RUN_TAG=$(date +%Y%m%d_%H%M)

case "$MODEL_INPUT" in
    "GT")
        MODEL_ALIAS="GT"
        MODEL="gt_sw"
        N_LAYERS=4
        HIDDEN_DIM=128
        FFN_DIM=128
        NUM_HEADS=8
        NPARTS=4096
        RELATED_TOPK=4
        ;;
    "GPH_Slim")
        MODEL_ALIAS="GPH_Slim"
        MODEL="graphormer"
        N_LAYERS=4
        HIDDEN_DIM=64
        FFN_DIM=64
        NUM_HEADS=8
        NPARTS=4096
        RELATED_TOPK=4
        ;;
    "GPH_Large")
        MODEL_ALIAS="GPH_Large"
        MODEL="graphormer"
        N_LAYERS=12
        HIDDEN_DIM=768
        FFN_DIM=768
        NUM_HEADS=32
        NPARTS=8192
        RELATED_TOPK=2
        ;;
esac

# 论文100M 专用 PPR 参数
#   ppr_batch_size=32768: 减少 SpGEMM 启动次数 (111M/32768≈3400 batch)
#   ppr_iter_topk=128:    每轮迭代剪枝，控制中间状态爆炸
ATTN_TYPE="sparse"
EPOCHS=500
PPR_BACKEND="appnp"
PPR_TOPK=5
PPR_ALPHA=0.85
PPR_NUM_ITER=10
PPR_BATCH_SIZE=32768
PPR_ITER_TOPK=128
USE_CACHE=1

# ==================== Main Execution ====================

IFS=, read -r -a GPU_LIST <<< "$DEVICES"
GPU_NUM=${#GPU_LIST[@]}

mkdir -p "${LOG_DIR}"

MODE_LABEL="train"
if [ "$PREPROCESS_ONLY" -eq 1 ]; then
    MODE_LABEL="preprocess"
fi

LOG_FILE="${LOG_DIR}/${DATASET}_${MODEL_ALIAS}_nparts${NPARTS}_rtopk${RELATED_TOPK}_${MODE_LABEL}_${RUN_TAG}.log"
MASTER_PORT=$((8000 + RANDOM % 1000))

echo "============================================================="
echo "  NeutronGT - papers100M Test"
echo "============================================================="
echo "  Dataset:       ${DATASET} (~111M nodes, ~1.6B edges)"
echo "  Model:         ${MODEL_ALIAS} (${MODEL})"
echo "  Config:        layers=${N_LAYERS} hidden=${HIDDEN_DIM} ffn=${FFN_DIM} heads=${NUM_HEADS}"
echo "  Windows:       n_parts=${NPARTS} related_topk=${RELATED_TOPK}"
echo "  PPR:           backend=${PPR_BACKEND} topk=${PPR_TOPK} batch=${PPR_BATCH_SIZE} iter_topk=${PPR_ITER_TOPK}"
echo "  GPUs:          ${GPU_NUM} (CUDA_VISIBLE_DEVICES=${DEVICES})"
echo "  Mode:          ${MODE_LABEL}"
echo "  Log:           ${LOG_FILE}"
echo "============================================================="
echo ""
echo "  显存预估 (per window, fp32):"

case "$MODEL_INPUT" in
    "GPH_Slim")
        echo "    ~31K nodes/window → ~1.1 GB GPU memory"
        ;;
    "GT")
        echo "    ~31K nodes/window → ~2.0 GB GPU memory"
        ;;
    "GPH_Large")
        echo "    ~16K nodes/window → ~9.5 GB GPU memory"
        ;;
esac
echo ""

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
    --distributed-timeout-minutes 180 \
    > "${LOG_FILE}" 2>&1

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Status: Success"
else
    echo "Status: Failed (exit code ${EXIT_CODE}). Check ${LOG_FILE}"
fi
