#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")/.."

export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

DATASET_DIR=./dataset/
LOG_DIR=./logs/window_sensitivity
MASTER_PORT=29950
RUN_TAG=$(date +%Y%m%d_%H%M%S)
EPOCHS=505
DATASETS=${1:-AmazonProducts,ogbn-products}

IFS=, read -r -a DATASET_LIST <<< "$DATASETS"
IFS=, read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
GPU_NUM=${#GPU_LIST[@]}

mkdir -p "$LOG_DIR"

echo "datasets=$DATASETS"
echo "model=Graphormer-Slim"
echo "attn_type=sparse"
echo "use_cache=1"
echo "epochs=$EPOCHS"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "log_dir=$LOG_DIR"

for dataset in "${DATASET_LIST[@]}"; do
  if [ "$dataset" = "ogbn-products" ]; then
    RELATED_TOPK=6
    NPARTS_LIST=(64 96 128 160 192)
  elif [ "$dataset" = "AmazonProducts" ]; then
    RELATED_TOPK=4
    NPARTS_LIST=(16 32 64 96 128 160)
  elif [ "$dataset" = "ogbn-arxiv" ]; then
    RELATED_TOPK=6
    NPARTS_LIST=(40 48 56 64 72)
  else
    echo "Unsupported dataset: $dataset" >&2
    exit 1
  fi

  for n_parts in "${NPARTS_LIST[@]}"; do
    log_file="$LOG_DIR/${dataset}__nparts${n_parts}__rtopk${RELATED_TOPK}__slim__cache1__${RUN_TAG}.log"

    echo "============================================================"
    echo "dataset=$dataset n_parts=$n_parts related_topk=$RELATED_TOPK"
    echo "log=$log_file"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun \
      --nproc_per_node="$GPU_NUM" \
      --master_port "$MASTER_PORT" \
      main_sp_node_level_ppr.py \
      --dataset "$dataset" \
      --dataset_dir "$DATASET_DIR" \
      --model graphormer \
      --n_layers 4 \
      --hidden_dim 64 \
      --ffn_dim 64 \
      --num_heads 8 \
      --epochs "$EPOCHS" \
      --struct_enc False \
      --use_preprocess_cache 0 \
      --attn_type sparse \
      --use_cache 1 \
      --n_parts "$n_parts" \
      --related_nodes_topk_rate "$RELATED_TOPK" \
      --ppr_backend appnp \
      --ppr_topk 5 \
      --ppr_alpha 0.85 \
      --ppr_num_iterations 10 \
      --ppr_batch_size 8192 \
      --ppr_iter_topk 5 \
      --distributed-backend nccl \
      --distributed-timeout-minutes 120 \
      2>&1 | tee "$log_file"

    MASTER_PORT=$((MASTER_PORT + 1))
  done
done
