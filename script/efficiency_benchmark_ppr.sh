#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES_LIST="${CUDA_VISIBLE_DEVICES_LIST:-0,1,2,3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29750}"
DATASET_DIR="${DATASET_DIR:-./dataset}"
LOG_DIR="${LOG_DIR:-./logs/fixed_ppr_25ep}"
RESULT_DIR="${RESULT_DIR:-./results/fixed_ppr_25ep}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DATASETS="${RUN_DATASETS:-all}"
RUN_MODELS="${RUN_MODELS:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GT_N_LAYERS="${GT_N_LAYERS:-4}"
GT_HIDDEN_DIM="${GT_HIDDEN_DIM:-128}"
GT_NUM_HEADS="${GT_NUM_HEADS:-8}"

GRAPHORMER_SLIM_N_LAYERS="${GRAPHORMER_SLIM_N_LAYERS:-4}"
GRAPHORMER_SLIM_HIDDEN_DIM="${GRAPHORMER_SLIM_HIDDEN_DIM:-64}"
GRAPHORMER_SLIM_NUM_HEADS="${GRAPHORMER_SLIM_NUM_HEADS:-8}"

GRAPHORMER_LARGE_N_LAYERS="${GRAPHORMER_LARGE_N_LAYERS:-12}"
GRAPHORMER_LARGE_HIDDEN_DIM="${GRAPHORMER_LARGE_HIDDEN_DIM:-768}"
GRAPHORMER_LARGE_NUM_HEADS="${GRAPHORMER_LARGE_NUM_HEADS:-32}"

EPOCHS=25
SEQ_LEN="${SEQ_LEN:-6400}"
STRUCT_ENC=False
MAX_DIST="${MAX_DIST:-5}"
USE_CACHE=1
PPR_BACKEND="${PPR_BACKEND:-appnp}"
PPR_TOPK="${PPR_TOPK:-5}"
PPR_ALPHA="${PPR_ALPHA:-0.85}"
PPR_NUM_ITERATIONS="${PPR_NUM_ITERATIONS:-10}"
PPR_BATCH_SIZE="${PPR_BATCH_SIZE:-8192}"
PPR_ITER_TOPK="${PPR_ITER_TOPK:-5}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-120}"

DATASETS=(
  "ogbn-arxiv"
  "reddit"
  "AmazonProducts"
  "ogbn-products"
)

MODEL_ALIASES=(
  "GT"
  "Graphormer-Slim"
  "Graphormer-Large"
)

DEVICE_COUNT=$(awk -F',' 'NF { print NF }' <<< "${CUDA_VISIBLE_DEVICES_LIST}")
[[ "$DATASET_DIR" != */ ]] && DATASET_DIR="${DATASET_DIR}/"
mkdir -p "$LOG_DIR" "$RESULT_DIR"
RESULT_CSV="${RESULT_DIR}/fixed_ppr_25ep_${RUN_TAG}.csv"

CSV_HEADER='run_tag,dataset,model_alias,model_cli,attn_type,n_layers,hidden_dim,ffn_dim,num_heads,n_parts,related_nodes_topk_rate,status,last_epoch,last_loss,last_train_time_s,last_cpu_to_gpu_time_s,last_window_fw_bw_avg_time_s,last_window_fw_bw_total_time_s,last_num_windows,last_train_acc,last_valid_acc,last_test_acc,log_path'

IFS=',' read -r -a RUN_DATASET_FILTER <<< "$RUN_DATASETS"
IFS=',' read -r -a RUN_MODEL_FILTER <<< "$RUN_MODELS"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

csv_escape() {
  local value="${1:-}"
  value=${value//\"/\"\"}
  printf '"%s"' "${value}"
}

append_csv_header() {
  if [[ -f "$RESULT_CSV" ]]; then
    return
  fi
  printf '%s\n' "$CSV_HEADER" > "$RESULT_CSV"
}

append_csv_row() {
  local row="$1"
  printf '%s\n' "$row" >> "$RESULT_CSV"
}

contains_exact() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

split_csv_to_array() {
  local csv="$1"
  local __resultvar="$2"
  local old_ifs="${IFS}"
  local -a parts=()
  local -n out_ref="${__resultvar}"
  IFS=',' read -r -a parts <<< "${csv}"
  IFS="${old_ifs}"
  out_ref=("${parts[@]}")
}

get_selected_datasets() {
  if [[ "$RUN_DATASETS" == "all" ]]; then
    printf '%s\n' "${DATASETS[@]}"
    return
  fi
  local -a selected=()
  split_csv_to_array "$RUN_DATASETS" selected
  printf '%s\n' "${selected[@]}"
}

get_selected_models() {
  if [[ "$RUN_MODELS" == "all" ]]; then
    printf '%s\n' "${MODEL_ALIASES[@]}"
    return
  fi
  local -a selected=()
  split_csv_to_array "$RUN_MODELS" selected
  printf '%s\n' "${selected[@]}"
}

get_model_args() {
  case "$1" in
    "GT")
      printf '%s|%s|%s|%s|%s|%s' "gt_sw" "sparse" "$GT_N_LAYERS" "$GT_HIDDEN_DIM" "$GT_NUM_HEADS" "$GT_HIDDEN_DIM"
      ;;
    "Graphormer-Slim")
      printf '%s|%s|%s|%s|%s|%s' "graphormer" "sparse" "$GRAPHORMER_SLIM_N_LAYERS" "$GRAPHORMER_SLIM_HIDDEN_DIM" "$GRAPHORMER_SLIM_NUM_HEADS" "$GRAPHORMER_SLIM_HIDDEN_DIM"
      ;;
    "Graphormer-Large")
      printf '%s|%s|%s|%s|%s|%s' "graphormer" "sparse" "$GRAPHORMER_LARGE_N_LAYERS" "$GRAPHORMER_LARGE_HIDDEN_DIM" "$GRAPHORMER_LARGE_NUM_HEADS" "$GRAPHORMER_LARGE_HIDDEN_DIM"
      ;;
    *)
      fail "Unknown model alias: $1"
      ;;
  esac
}

get_fixed_params() {
  case "$1:$2" in
    "ogbn-arxiv:GT") printf '16|4' ;;
    "ogbn-arxiv:Graphormer-Slim") printf '20|4' ;;
    "ogbn-arxiv:Graphormer-Large") printf '10|4' ;;

    "reddit:GT") printf '16|4' ;;
    "reddit:Graphormer-Slim") printf '20|4' ;;
    "reddit:Graphormer-Large") printf '10|4' ;;

    "AmazonProducts:GT") printf '64|2' ;;
    "AmazonProducts:Graphormer-Slim") printf '64|2' ;;
    "AmazonProducts:Graphormer-Large") printf '128|2' ;;

    "ogbn-products:GT") printf '128|2' ;;
    "ogbn-products:Graphormer-Slim") printf '128|2' ;;
    "ogbn-products:Graphormer-Large") printf '256|2' ;;
    *) fail "No fixed parameters for dataset/model: $1 / $2" ;;
  esac
}

extract_last_train_line() {
  local log_file="$1"
  grep "Train Time:" "$log_file" | tail -n 1 || true
}

extract_last_eval_line() {
  local log_file="$1"
  grep "train_acc:" "$log_file" | tail -n 1 || true
}

parse_value_by_regex() {
  local line="$1"
  local regex="$2"
  sed -nE "s|${regex}|\1|p" <<< "${line}"
}

append_failed_row() {
  local dataset="$1"
  local model_alias="$2"
  local model_cli="$3"
  local attn_type="$4"
  local n_layers="$5"
  local hidden_dim="$6"
  local ffn_dim="$7"
  local num_heads="$8"
  local n_parts="$9"
  local related_topk="${10}"
  local log_file="${11}"
  local row
  row="$({
    printf '%s,' "$(csv_escape "$RUN_TAG")"
    printf '%s,' "$(csv_escape "$dataset")"
    printf '%s,' "$(csv_escape "$model_alias")"
    printf '%s,' "$(csv_escape "$model_cli")"
    printf '%s,' "$(csv_escape "$attn_type")"
    printf '%s,' "$(csv_escape "$n_layers")"
    printf '%s,' "$(csv_escape "$hidden_dim")"
    printf '%s,' "$(csv_escape "$ffn_dim")"
    printf '%s,' "$(csv_escape "$num_heads")"
    printf '%s,' "$(csv_escape "$n_parts")"
    printf '%s,' "$(csv_escape "$related_topk")"
    printf '%s,' "$(csv_escape "FAILED")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s' "$(csv_escape "$log_file")"
  })"
  append_csv_row "$row"
}

append_success_row() {
  local dataset="$1"
  local model_alias="$2"
  local model_cli="$3"
  local attn_type="$4"
  local n_layers="$5"
  local hidden_dim="$6"
  local ffn_dim="$7"
  local num_heads="$8"
  local n_parts="$9"
  local related_topk="${10}"
  local log_file="${11}"

  local last_train_line
  local last_eval_line
  local last_epoch
  local last_loss
  local last_train_time
  local last_cpu_to_gpu_time
  local last_window_fw_bw_avg_time
  local last_window_fw_bw_total_time
  local last_num_windows
  local last_train_acc
  local last_valid_acc
  local last_test_acc
  local row

  last_train_line="$(extract_last_train_line "$log_file")"
  last_eval_line="$(extract_last_eval_line "$log_file")"

  last_epoch="$(parse_value_by_regex "$last_train_line" '.*Epoch: ([0-9]+), Loss: .*')"
  last_loss="$(parse_value_by_regex "$last_train_line" '.*Loss: ([0-9.]+), Train Time: .*')"
  last_train_time="$(parse_value_by_regex "$last_train_line" '.*Train Time: ([0-9.]+)s, CPU->GPU Time: .*')"
  last_cpu_to_gpu_time="$(parse_value_by_regex "$last_train_line" '.*CPU->GPU Time: ([0-9.]+)s, Window FW/BW Avg: .*')"
  last_window_fw_bw_avg_time="$(parse_value_by_regex "$last_train_line" '.*Window FW/BW Avg: ([0-9.]+)s, Window FW/BW Total: .*')"
  last_window_fw_bw_total_time="$(parse_value_by_regex "$last_train_line" '.*Window FW/BW Total: ([0-9.]+)s, Windows: .*')"
  last_num_windows="$(parse_value_by_regex "$last_train_line" '.*Windows: ([0-9]+), Max Window Steps: .*')"

  last_train_acc="$(parse_value_by_regex "$last_eval_line" '.*train_acc: ([0-9.]+)%, valid_acc: .*')"
  last_valid_acc="$(parse_value_by_regex "$last_eval_line" '.*valid_acc: ([0-9.]+)%, test_acc: .*')"
  last_test_acc="$(parse_value_by_regex "$last_eval_line" '.*test_acc: ([0-9.]+)%, samples.*')"

  row="$({
    printf '%s,' "$(csv_escape "$RUN_TAG")"
    printf '%s,' "$(csv_escape "$dataset")"
    printf '%s,' "$(csv_escape "$model_alias")"
    printf '%s,' "$(csv_escape "$model_cli")"
    printf '%s,' "$(csv_escape "$attn_type")"
    printf '%s,' "$(csv_escape "$n_layers")"
    printf '%s,' "$(csv_escape "$hidden_dim")"
    printf '%s,' "$(csv_escape "$ffn_dim")"
    printf '%s,' "$(csv_escape "$num_heads")"
    printf '%s,' "$(csv_escape "$n_parts")"
    printf '%s,' "$(csv_escape "$related_topk")"
    printf '%s,' "$(csv_escape "SUCCESS")"
    printf '%s,' "$(csv_escape "$last_epoch")"
    printf '%s,' "$(csv_escape "$last_loss")"
    printf '%s,' "$(csv_escape "$last_train_time")"
    printf '%s,' "$(csv_escape "$last_cpu_to_gpu_time")"
    printf '%s,' "$(csv_escape "$last_window_fw_bw_avg_time")"
    printf '%s,' "$(csv_escape "$last_window_fw_bw_total_time")"
    printf '%s,' "$(csv_escape "$last_num_windows")"
    printf '%s,' "$(csv_escape "$last_train_acc")"
    printf '%s,' "$(csv_escape "$last_valid_acc")"
    printf '%s,' "$(csv_escape "$last_test_acc")"
    printf '%s' "$(csv_escape "$log_file")"
  })"
  append_csv_row "$row"
}

validate_user_inputs() {
  [[ "$DEVICE_COUNT" -ge 1 ]] || fail "CUDA_VISIBLE_DEVICES_LIST must contain at least one device."

  local -a selected_datasets=()
  local -a selected_models=()
  mapfile -t selected_datasets < <(get_selected_datasets)
  mapfile -t selected_models < <(get_selected_models)

  local dataset
  for dataset in "${selected_datasets[@]}"; do
    contains_exact "$dataset" "${DATASETS[@]}" || fail "Unknown dataset in RUN_DATASETS: $dataset"
  done

  local model_alias
  for model_alias in "${selected_models[@]}"; do
    contains_exact "$model_alias" "${MODEL_ALIASES[@]}" || fail "Unknown model in RUN_MODELS: $model_alias"
  done
}

run_one_experiment() {
  local dataset="$1"
  local model_alias="$2"
  local port="$3"
  local model_cli attn_type n_layers hidden_dim num_heads ffn_dim
  local n_parts related_topk log_file command_status
  local -a cmd

  IFS='|' read -r model_cli attn_type n_layers hidden_dim num_heads ffn_dim <<< "$(get_model_args "$model_alias")"
  IFS='|' read -r n_parts related_topk <<< "$(get_fixed_params "$dataset" "$model_alias")"

  log_file="${LOG_DIR}/${dataset}__${model_alias}__${RUN_TAG}.log"
  cmd=(
    torchrun
    --nproc_per_node="${DEVICE_COUNT}"
    --master_port "${port}"
    main_sp_node_level_ppr.py
    --dataset "$dataset"
    --dataset_dir "$DATASET_DIR"
    --seq_len "$SEQ_LEN"
    --n_layers "$n_layers"
    --hidden_dim "$hidden_dim"
    --ffn_dim "$ffn_dim"
    --num_heads "$num_heads"
    --epochs "$EPOCHS"
    --model "$model_cli"
    --distributed-backend nccl
    --attn_type "$attn_type"
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
  )

  echo "===================================================================================="
  echo "[RUN] dataset=${dataset} model=${model_alias} n_parts=${n_parts} related_topk=${related_topk} port=${port}"
  echo "[RUN] log=${log_file}"
  echo "===================================================================================="

  set +e
  {
    printf '# run_tag=%s\n' "$RUN_TAG"
    printf '# dataset=%s\n' "$dataset"
    printf '# model_alias=%s\n' "$model_alias"
    printf '# n_parts=%s\n' "$n_parts"
    printf '# related_nodes_topk_rate=%s\n' "$related_topk"
    printf '# command='
    printf '%q ' env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}" "${cmd[@]}"
    printf '\n'
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_LIST" "${cmd[@]}"
  } 2>&1 | tee "$log_file"
  command_status=${PIPESTATUS[0]}
  set -e

  if [[ "$command_status" -ne 0 ]]; then
    echo "[WARN] Experiment failed: dataset=${dataset}, model=${model_alias}, status=${command_status}" >&2
    append_failed_row "$dataset" "$model_alias" "$model_cli" "$attn_type" "$n_layers" "$hidden_dim" "$ffn_dim" "$num_heads" "$n_parts" "$related_topk" "$log_file"
    return 0
  fi

  append_success_row "$dataset" "$model_alias" "$model_cli" "$attn_type" "$n_layers" "$hidden_dim" "$ffn_dim" "$num_heads" "$n_parts" "$related_topk" "$log_file"
}

main() {
  local current_port
  local dataset
  local model_alias
  local -a selected_datasets=()
  local -a selected_models=()

  validate_user_inputs
  append_csv_header

  mapfile -t selected_datasets < <(get_selected_datasets)
  mapfile -t selected_models < <(get_selected_models)

  echo "===================================================================================="
  echo "[CONFIG] datasets: ${selected_datasets[*]}"
  echo "[CONFIG] models: ${selected_models[*]}"
  echo "[CONFIG] epochs: ${EPOCHS}"
  echo "[CONFIG] struct_enc: ${STRUCT_ENC}"
  echo "[CONFIG] use_cache: ${USE_CACHE}"
  echo "===================================================================================="

  current_port="$MASTER_PORT_BASE"
  for dataset in "${selected_datasets[@]}"; do
    for model_alias in "${selected_models[@]}"; do
      run_one_experiment "$dataset" "$model_alias" "$current_port"
      current_port=$((current_port + 1))
    done
  done

  echo "===================================================================================="
  echo "[DONE] Fixed 25-epoch runs finished."
  echo "[DONE] CSV summary: ${RESULT_CSV}"
  echo "===================================================================================="
}

main "$@"
