#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")/.."

export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET_DIR=./dataset/
LOG_DIR=./full_attention_window_size_logs/search_window_size_graphormer_slim
MASTER_PORT=29660
RUN_TAG=$(date +%Y%m%d_%H%M%S)
DATASETS=${1:-ogbn-products}

IFS=, read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
GPU_NUM=${#GPU_LIST[@]}
IFS=, read -r -a DATASET_LIST <<< "$DATASETS"

mkdir -p "$LOG_DIR"

echo "datasets=$DATASETS"
echo "model=Graphormer-Slim"
echo "search_mode=full_attention_1epoch_stop_on_first_success"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "log_dir=$LOG_DIR"

for dataset in "${DATASET_LIST[@]}"; do
  if [ "$dataset" = "ogbn-arxiv" ]; then
    RELATED_TOPK=8
    CANDIDATE_NPARTS="56"
  elif [ "$dataset" = "reddit" ]; then
    RELATED_TOPK=10
    CANDIDATE_NPARTS="64,80,96"
  elif [ "$dataset" = "AmazonProducts" ]; then
    RELATED_TOPK=4
    CANDIDATE_NPARTS="256,320"
  elif [ "$dataset" = "ogbn-products" ]; then
    RELATED_TOPK=6
    CANDIDATE_NPARTS="600,640"
  else
    echo "Unsupported dataset: $dataset" >&2
    exit 1
  fi

  IFS=, read -r -a NPARTS_LIST <<< "$CANDIDATE_NPARTS"

  echo "======================================================================"
  echo "dataset=$dataset related_topk=$RELATED_TOPK candidate_nparts=$CANDIDATE_NPARTS"
  echo "======================================================================"

  for n_parts in "${NPARTS_LIST[@]}"; do
    log_file="$LOG_DIR/${dataset}__full__nparts${n_parts}__${RUN_TAG}.log"

    echo "------------------------------------------------------------"
    echo "dataset=$dataset n_parts=$n_parts attn_type=full use_cache=0"
    echo "log=$log_file"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun \
      --nproc_per_node="$GPU_NUM" \
      --master_port "$MASTER_PORT" \
      main_sp_node_level_ppr.py \
      --dataset "$dataset" \
      --dataset_dir "$DATASET_DIR" \
      --model graphormer \
      --attn_type full \
      --n_layers 4 \
      --hidden_dim 64 \
      --ffn_dim 64 \
      --num_heads 8 \
      --epochs 25 \
      --seq_len 64000 \
      --struct_enc False \
      --max_dist 5 \
      --use_cache 0 \
      --num_global_node 1 \
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

    status=${PIPESTATUS[0]}
    MASTER_PORT=$((MASTER_PORT + 1))

    if [ "$status" -eq 0 ]; then
      echo "[OK] dataset=$dataset first_success_nparts=$n_parts" | tee -a "$log_file"
      break
    else
      echo "[FAILED] dataset=$dataset n_parts=$n_parts status=$status" | tee -a "$log_file"
    fi
  done
done
