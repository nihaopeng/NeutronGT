#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash script/run_accuracy.sh
#   bash script/run_accuracy.sh reddit
#   bash script/run_accuracy.sh ogbn-products Graphormer-Slim
#
# Args:
#   $1: datasets (optional, default: all) one of: reddit, ogbn-products, all
#   $2: models   (optional, default: all) one of: GT, Graphormer-Slim, Graphormer-Large, all

CUDA_VISIBLE_DEVICES_LIST="${CUDA_VISIBLE_DEVICES_LIST:-0,1,2,3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29850}"
RUN_DATASETS="${1:-all}"
RUN_MODELS="${2:-all}"

IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES_LIST"
DEVICE_NUM=${#DEVICES[@]}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-./dataset/}"
LOG_DIR="${LOG_DIR:-./logs/accuracy_simple}"
RESULT_DIR="${RESULT_DIR:-./results/accuracy_simple}"
CHECKPOINT_ROOT_DIR="${CHECKPOINT_ROOT_DIR:-./model_ckpt/accuracy_simple}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

STRUCT_ENC="${STRUCT_ENC:-False}"
USE_CACHE="${USE_CACHE:-1}"
PPR_BACKEND="${PPR_BACKEND:-appnp}"
PPR_TOPK="${PPR_TOPK:-5}"
PPR_ALPHA="${PPR_ALPHA:-0.85}"
PPR_NUM_ITERATIONS="${PPR_NUM_ITERATIONS:-10}"
PPR_BATCH_SIZE="${PPR_BATCH_SIZE:-8192}"
PPR_ITER_TOPK="${PPR_ITER_TOPK:-5}"
SEQ_LEN="${SEQ_LEN:-6400}"
MAX_DIST="${MAX_DIST:-5}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-120}"
NUM_GLOBAL_NODE="${NUM_GLOBAL_NODE:-1}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$CHECKPOINT_ROOT_DIR"
RESULT_CSV="$RESULT_DIR/accuracy_simple_${RUN_TAG}.csv"
CSV_HEADER='run_tag,dataset,model_alias,epochs,n_parts,related_nodes_topk_rate,status,log_path,checkpoint_dir'

DATASETS=("reddit" "ogbn-products")
MODELS=("GT" "Graphormer-Slim" "Graphormer-Large")

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

csv_escape() {
  local value="${1:-}"
  local escaped
  escaped=$(printf '%s' "$value" | sed 's/"/""/g')
  printf '"%s"' "$escaped"
}

append_csv_header() {
  if [[ ! -f "$RESULT_CSV" ]]; then
    printf '%s\n' "$CSV_HEADER" > "$RESULT_CSV"
  fi
}

append_csv_row() {
  printf '%s\n' "$1" >> "$RESULT_CSV"
}

contains_exact() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

split_csv_to_array() {
  local csv="$1"
  local __resultvar="$2"
  local -n out_ref="$__resultvar"
  IFS=',' read -r -a out_ref <<< "$csv"
}

get_selected_datasets() {
  if [[ "$RUN_DATASETS" == "all" ]]; then
    printf '%s\n' "${DATASETS[@]}"
  else
    local -a selected=()
    split_csv_to_array "$RUN_DATASETS" selected
    printf '%s\n' "${selected[@]}"
  fi
}

get_selected_models() {
  if [[ "$RUN_MODELS" == "all" ]]; then
    printf '%s\n' "${MODELS[@]}"
  else
    local -a selected=()
    split_csv_to_array "$RUN_MODELS" selected
    printf '%s\n' "${selected[@]}"
  fi
}

get_model_args() {
  case "$1" in
    "GT") printf 'gt_sw|500|4|128|8|128' ;;
    "Graphormer-Slim") printf 'graphormer|500|4|64|8|64' ;;
    "Graphormer-Large") printf 'graphormer|200|12|768|32|768' ;;
    *) fail "Unknown model alias: $1" ;;
  esac
}

# ===== Edit these 6 lines when you want to change the experiment params =====
get_case_params() {
  case "$1:$2" in
    "reddit:GT") printf '32|10' ;;
    "reddit:Graphormer-Slim") printf '32|10' ;;
    "reddit:Graphormer-Large") printf '80|4' ;;
    "ogbn-products:GT") printf '128|6' ;;
    "ogbn-products:Graphormer-Slim") printf '128|6' ;;
    "ogbn-products:Graphormer-Large") printf '512|4' ;;
    *) fail "No params for dataset/model: $1 / $2" ;;
  esac
}
# ==========================================================================

append_status_row() {
  local dataset="$1"
  local model_alias="$2"
  local epochs="$3"
  local n_parts="$4"
  local related_topk="$5"
  local status="$6"
  local log_path="$7"
  local checkpoint_dir="$8"
  local row
  row="$({
    printf '%s,' "$(csv_escape "$RUN_TAG")"
    printf '%s,' "$(csv_escape "$dataset")"
    printf '%s,' "$(csv_escape "$model_alias")"
    printf '%s,' "$(csv_escape "$epochs")"
    printf '%s,' "$(csv_escape "$n_parts")"
    printf '%s,' "$(csv_escape "$related_topk")"
    printf '%s,' "$(csv_escape "$status")"
    printf '%s,' "$(csv_escape "$log_path")"
    printf '%s' "$(csv_escape "$checkpoint_dir")"
  })"
  append_csv_row "$row"
}

validate_inputs() {
  [[ "$DEVICE_NUM" -ge 1 ]] || fail "No CUDA devices provided."
  local -a selected_datasets=() selected_models=()
  local dataset model_alias
  mapfile -t selected_datasets < <(get_selected_datasets)
  mapfile -t selected_models < <(get_selected_models)
  for dataset in "${selected_datasets[@]}"; do
    contains_exact "$dataset" "${DATASETS[@]}" || fail "Unknown dataset: $dataset"
  done
  for model_alias in "${selected_models[@]}"; do
    contains_exact "$model_alias" "${MODELS[@]}" || fail "Unknown model: $model_alias"
  done
}

run_one() {
  local dataset="$1"
  local model_alias="$2"
  local port="$3"
  local model_cli epochs n_layers hidden_dim num_heads ffn_dim
  local n_parts related_topk checkpoint_dir log_file status
  local -a cmd

  IFS='|' read -r model_cli epochs n_layers hidden_dim num_heads ffn_dim <<< "$(get_model_args "$model_alias")"
  IFS='|' read -r n_parts related_topk <<< "$(get_case_params "$dataset" "$model_alias")"

  checkpoint_dir="${CHECKPOINT_ROOT_DIR}/${dataset}/${model_alias}/"
  log_file="${LOG_DIR}/${dataset}__${model_alias}__${RUN_TAG}.log"

  cmd=(
    torchrun
    --nproc_per_node="$DEVICE_NUM"
    --master_port "$port"
    main_sp_node_level_ppr.py
    --dataset "$dataset"
    --dataset_dir "$DATASET_DIR"
    --seq_len "$SEQ_LEN"
    --n_layers "$n_layers"
    --hidden_dim "$hidden_dim"
    --ffn_dim "$ffn_dim"
    --num_heads "$num_heads"
    --epochs "$epochs"
    --model "$model_cli"
    --distributed-backend nccl
    --attn_type sparse
    --struct_enc "$STRUCT_ENC"
    --max_dist "$MAX_DIST"
    --use_cache "$USE_CACHE"
    --n_parts "$n_parts"
    --related_nodes_topk_rate "$related_topk"
    --ppr_backend "$PPR_BACKEND"
    --ppr_topk "$PPR_TOPK"
    --ppr_alpha "$PPR_ALPHA"
    --ppr_num_iterations "$PPR_NUM_ITERATIONS"
    --ppr_batch_size "$PPR_BATCH_SIZE"
    --ppr_iter_topk "$PPR_ITER_TOPK"
    --distributed-timeout-minutes "$DISTRIBUTED_TIMEOUT_MINUTES"
    --num_global_node "$NUM_GLOBAL_NODE"
    --save_checkpoint
    --resume_latest
    --save_latest_only
    --checkpoint_dir "$checkpoint_dir"
  )

  echo "===================================================================================="
  echo "[RUN] dataset=${dataset} model=${model_alias} epochs=${epochs} n_parts=${n_parts} related_topk=${related_topk} port=${port}"
  echo "[RUN] checkpoint_dir=${checkpoint_dir}"
  echo "[RUN] log=${log_file}"
  echo "===================================================================================="

  set +e
  env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_LIST" "${cmd[@]}" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[WARN] failed: dataset=${dataset} model=${model_alias} status=${status}" >&2
    append_status_row "$dataset" "$model_alias" "$epochs" "$n_parts" "$related_topk" "FAILED" "$log_file" "$checkpoint_dir"
    return 0
  fi

  append_status_row "$dataset" "$model_alias" "$epochs" "$n_parts" "$related_topk" "SUCCESS" "$log_file" "$checkpoint_dir"
}

main() {
  local current_port dataset model_alias
  local -a selected_datasets=() selected_models=()

  validate_inputs
  append_csv_header
  mapfile -t selected_datasets < <(get_selected_datasets)
  mapfile -t selected_models < <(get_selected_models)

  echo "[CONFIG] datasets: ${selected_datasets[*]}"
  echo "[CONFIG] models: ${selected_models[*]}"
  echo "[CONFIG] struct_enc=${STRUCT_ENC} use_cache=${USE_CACHE} ppr_backend=${PPR_BACKEND}"

  current_port="$MASTER_PORT_BASE"
  for dataset in "${selected_datasets[@]}"; do
    for model_alias in "${selected_models[@]}"; do
      run_one "$dataset" "$model_alias" "$current_port"
      current_port=$((current_port + 1))
    done
  done

  echo "[DONE] csv: $RESULT_CSV"
}

main
