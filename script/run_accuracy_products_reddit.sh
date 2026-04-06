#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES_LIST="${CUDA_VISIBLE_DEVICES_LIST:-0,1,2,3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29850}"
DATASET_DIR="${DATASET_DIR:-./dataset}"
LOG_DIR="${LOG_DIR:-./logs/accuracy_products_reddit}"
RESULT_DIR="${RESULT_DIR:-./results/accuracy_products_reddit}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DATASETS="${RUN_DATASETS:-all}"
RUN_MODELS="${RUN_MODELS:-all}"
CHECKPOINT_ROOT_DIR="${CHECKPOINT_ROOT_DIR:-./model_ckpt/accuracy_products_reddit}"

GT_N_LAYERS="${GT_N_LAYERS:-4}"
GT_HIDDEN_DIM="${GT_HIDDEN_DIM:-128}"
GT_NUM_HEADS="${GT_NUM_HEADS:-8}"
GRAPHORMER_SLIM_N_LAYERS="${GRAPHORMER_SLIM_N_LAYERS:-4}"
GRAPHORMER_SLIM_HIDDEN_DIM="${GRAPHORMER_SLIM_HIDDEN_DIM:-64}"
GRAPHORMER_SLIM_NUM_HEADS="${GRAPHORMER_SLIM_NUM_HEADS:-8}"
GRAPHORMER_LARGE_N_LAYERS="${GRAPHORMER_LARGE_N_LAYERS:-12}"
GRAPHORMER_LARGE_HIDDEN_DIM="${GRAPHORMER_LARGE_HIDDEN_DIM:-768}"
GRAPHORMER_LARGE_NUM_HEADS="${GRAPHORMER_LARGE_NUM_HEADS:-32}"

SEQ_LEN="${SEQ_LEN:-6400}"
STRUCT_ENC="${STRUCT_ENC:-False}"
MAX_DIST="${MAX_DIST:-5}"
USE_CACHE="${USE_CACHE:-1}"
PPR_BACKEND="${PPR_BACKEND:-appnp}"
PPR_TOPK="${PPR_TOPK:-5}"
PPR_ALPHA="${PPR_ALPHA:-0.85}"
PPR_NUM_ITERATIONS="${PPR_NUM_ITERATIONS:-10}"
PPR_BATCH_SIZE="${PPR_BATCH_SIZE:-8192}"
PPR_ITER_TOPK="${PPR_ITER_TOPK:-5}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-120}"
DROPOUT_RATE="${DROPOUT_RATE:-0.0}"
ATTN_DROPOUT_RATE="${ATTN_DROPOUT_RATE:-0.0}"
INPUT_DROPOUT_RATE="${INPUT_DROPOUT_RATE:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
PEAK_LR="${PEAK_LR:-0.001}"
END_LR="${END_LR:-1e-9}"
WARMUP_UPDATES="${WARMUP_UPDATES:-0}"
PATIENCE="${PATIENCE:-50}"
NUM_GLOBAL_NODE="${NUM_GLOBAL_NODE:-1}"

DATASETS=(
  "reddit"
  "ogbn-products"
)

MODEL_ALIASES=(
  "GT"
  "Graphormer-Slim"
  "Graphormer-Large"
)

DEVICE_COUNT=$(awk -F',' 'NF { print NF }' <<< "$CUDA_VISIBLE_DEVICES_LIST")
[[ "$DATASET_DIR" != */ ]] && DATASET_DIR="${DATASET_DIR}/"
[[ "$CHECKPOINT_ROOT_DIR" != */ ]] && CHECKPOINT_ROOT_DIR="${CHECKPOINT_ROOT_DIR}/"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$CHECKPOINT_ROOT_DIR"
RESULT_CSV="${RESULT_DIR}/accuracy_products_reddit_${RUN_TAG}.csv"
CSV_HEADER='run_tag,dataset,model_alias,model_cli,attn_type,epochs,n_layers,hidden_dim,ffn_dim,num_heads,n_parts,related_nodes_topk_rate,status,last_epoch,last_loss,last_train_time_s,last_cpu_to_gpu_time_s,last_window_fw_bw_avg_time_s,last_window_fw_bw_total_time_s,last_num_windows,last_train_acc,last_valid_acc,last_test_acc,checkpoint_dir,log_path'

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
  local old_ifs="$IFS"
  local -a parts=()
  local -n out_ref="$__resultvar"
  IFS=',' read -r -a parts <<< "$csv"
  IFS="$old_ifs"
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
      printf '%s|%s|%s|%s|%s|%s|%s' "gt_sw" "sparse" "500" "$GT_N_LAYERS" "$GT_HIDDEN_DIM" "$GT_NUM_HEADS" "$GT_HIDDEN_DIM"
      ;;
    "Graphormer-Slim")
      printf '%s|%s|%s|%s|%s|%s|%s' "graphormer" "sparse" "500" "$GRAPHORMER_SLIM_N_LAYERS" "$GRAPHORMER_SLIM_HIDDEN_DIM" "$GRAPHORMER_SLIM_NUM_HEADS" "$GRAPHORMER_SLIM_HIDDEN_DIM"
      ;;
    "Graphormer-Large")
      printf '%s|%s|%s|%s|%s|%s|%s' "graphormer" "sparse" "200" "$GRAPHORMER_LARGE_N_LAYERS" "$GRAPHORMER_LARGE_HIDDEN_DIM" "$GRAPHORMER_LARGE_NUM_HEADS" "$GRAPHORMER_LARGE_HIDDEN_DIM"
      ;;
    *)
      fail "Unknown model alias: $1"
      ;;
  esac
}

case_env_prefix() {
  local dataset="$1"
  local model_alias="$2"
  local dataset_prefix model_prefix

  case "$dataset" in
    "reddit") dataset_prefix="REDDIT" ;;
    "ogbn-products") dataset_prefix="OGBN_PRODUCTS" ;;
    *) fail "Unsupported dataset: $dataset" ;;
  esac

  case "$model_alias" in
    "GT") model_prefix="GT" ;;
    "Graphormer-Slim") model_prefix="GRAPHORMER_SLIM" ;;
    "Graphormer-Large") model_prefix="GRAPHORMER_LARGE" ;;
    *) fail "Unsupported model alias: $model_alias" ;;
  esac

  printf '%s_%s' "$dataset_prefix" "$model_prefix"
}

get_case_params() {
  local dataset="$1"
  local model_alias="$2"
  local prefix nparts_var topk_var nparts_val topk_val

  prefix=$(case_env_prefix "$dataset" "$model_alias")
  nparts_var="${prefix}_NPARTS"
  topk_var="${prefix}_RELATED_TOPK"
  nparts_val="${!nparts_var:-}"
  topk_val="${!topk_var:-}"

  [[ -n "$nparts_val" ]] || fail "Missing required env var: $nparts_var"
  [[ -n "$topk_val" ]] || fail "Missing required env var: $topk_var"

  printf '%s|%s' "$nparts_val" "$topk_val"
}

extract_last_train_line() {
  local log_file="$1"
  grep 'Train Time:' "$log_file" | tail -n 1 || true
}

extract_last_eval_line() {
  local log_file="$1"
  grep 'train_acc:' "$log_file" | tail -n 1 || true
}

parse_value_by_regex() {
  local line="$1"
  local regex="$2"
  sed -nE "s|${regex}|\\1|p" <<< "$line"
}

append_failed_row() {
  local dataset="$1"
  local model_alias="$2"
  local model_cli="$3"
  local attn_type="$4"
  local epochs="$5"
  local n_layers="$6"
  local hidden_dim="$7"
  local ffn_dim="$8"
  local num_heads="$9"
  local n_parts="${10}"
  local related_topk="${11}"
  local checkpoint_dir="${12}"
  local log_file="${13}"
  local row

  row="$({
    printf '%s,' "$(csv_escape "$RUN_TAG")"
    printf '%s,' "$(csv_escape "$dataset")"
    printf '%s,' "$(csv_escape "$model_alias")"
    printf '%s,' "$(csv_escape "$model_cli")"
    printf '%s,' "$(csv_escape "$attn_type")"
    printf '%s,' "$(csv_escape "$epochs")"
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
    printf '%s,' "$(csv_escape "$checkpoint_dir")"
    printf '%s' "$(csv_escape "$log_file")"
  })"
  append_csv_row "$row"
}

append_success_row() {
  local dataset="$1"
  local model_alias="$2"
  local model_cli="$3"
  local attn_type="$4"
  local epochs="$5"
  local n_layers="$6"
  local hidden_dim="$7"
  local ffn_dim="$8"
  local num_heads="$9"
  local n_parts="${10}"
  local related_topk="${11}"
  local checkpoint_dir="${12}"
  local log_file="${13}"

  local last_train_line last_eval_line last_epoch last_loss
  local last_train_time last_cpu_to_gpu_time last_window_fw_bw_avg_time
  local last_window_fw_bw_total_time last_num_windows last_train_acc
  local last_valid_acc last_test_acc row

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
    printf '%s,' "$(csv_escape "$epochs")"
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
    printf '%s,' "$(csv_escape "$checkpoint_dir")"
    printf '%s' "$(csv_escape "$log_file")"
  })"
  append_csv_row "$row"
}

validate_user_inputs() {
  [[ "$DEVICE_COUNT" -ge 1 ]] || fail 'CUDA_VISIBLE_DEVICES_LIST must contain at least one device.'

  local -a selected_datasets=()
  local -a selected_models=()
  local dataset model_alias

  mapfile -t selected_datasets < <(get_selected_datasets)
  mapfile -t selected_models < <(get_selected_models)

  for dataset in "${selected_datasets[@]}"; do
    contains_exact "$dataset" "${DATASETS[@]}" || fail "Unknown dataset in RUN_DATASETS: $dataset"
  done
  for model_alias in "${selected_models[@]}"; do
    contains_exact "$model_alias" "${MODEL_ALIASES[@]}" || fail "Unknown model in RUN_MODELS: $model_alias"
  done
}

run_one_experiment() {
  local dataset="$1"
  local model_alias="$2"
  local port="$3"
  local model_cli attn_type epochs n_layers hidden_dim num_heads ffn_dim
  local n_parts related_topk checkpoint_dir log_file command_status
  local -a cmd

  IFS='|' read -r model_cli attn_type epochs n_layers hidden_dim num_heads ffn_dim <<< "$(get_model_args "$model_alias")"
  IFS='|' read -r n_parts related_topk <<< "$(get_case_params "$dataset" "$model_alias")"

  checkpoint_dir="${CHECKPOINT_ROOT_DIR}${dataset}/${model_alias}/"
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
    --epochs "$epochs"
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
    --dropout_rate "$DROPOUT_RATE"
    --attention_dropout_rate "$ATTN_DROPOUT_RATE"
    --input_dropout_rate "$INPUT_DROPOUT_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --peak_lr "$PEAK_LR"
    --end_lr "$END_LR"
    --warmup_updates "$WARMUP_UPDATES"
    --patience "$PATIENCE"
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
  {
    printf '# run_tag=%s\n' "$RUN_TAG"
    printf '# dataset=%s\n' "$dataset"
    printf '# model_alias=%s\n' "$model_alias"
    printf '# epochs=%s\n' "$epochs"
    printf '# n_parts=%s\n' "$n_parts"
    printf '# related_nodes_topk_rate=%s\n' "$related_topk"
    printf '# checkpoint_dir=%s\n' "$checkpoint_dir"
    printf '# command='
    printf '%q ' env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}" "${cmd[@]}"
    printf '\n'
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_LIST" "${cmd[@]}"
  } 2>&1 | tee "$log_file"
  command_status=${PIPESTATUS[0]}
  set -e

  if [[ "$command_status" -ne 0 ]]; then
    echo "[WARN] Experiment failed: dataset=${dataset}, model=${model_alias}, status=${command_status}" >&2
    append_failed_row "$dataset" "$model_alias" "$model_cli" "$attn_type" "$epochs" "$n_layers" "$hidden_dim" "$ffn_dim" "$num_heads" "$n_parts" "$related_topk" "$checkpoint_dir" "$log_file"
    return 0
  fi

  append_success_row "$dataset" "$model_alias" "$model_cli" "$attn_type" "$epochs" "$n_layers" "$hidden_dim" "$ffn_dim" "$num_heads" "$n_parts" "$related_topk" "$checkpoint_dir" "$log_file"
}

main() {
  local current_port dataset model_alias
  local -a selected_datasets=()
  local -a selected_models=()

  validate_user_inputs
  mkdir -p "$LOG_DIR" "$RESULT_DIR"
  append_csv_header
  mapfile -t selected_datasets < <(get_selected_datasets)
  mapfile -t selected_models < <(get_selected_models)

  echo "===================================================================================="
  echo "[CONFIG] datasets: ${selected_datasets[*]}"
  echo "[CONFIG] models: ${selected_models[*]}"
  echo "[CONFIG] struct_enc: ${STRUCT_ENC}"
  echo "[CONFIG] use_cache: ${USE_CACHE}"
  echo "[CONFIG] checkpoint root: ${CHECKPOINT_ROOT_DIR}"
  echo "===================================================================================="

  current_port="$MASTER_PORT_BASE"
  for dataset in "${selected_datasets[@]}"; do
    for model_alias in "${selected_models[@]}"; do
      run_one_experiment "$dataset" "$model_alias" "$current_port"
      current_port=$((current_port + 1))
    done
  done

  echo "===================================================================================="
  echo "[DONE] Accuracy runs finished."
  echo "[DONE] CSV summary: ${RESULT_CSV}"
  echo "===================================================================================="
}

main "$@"
