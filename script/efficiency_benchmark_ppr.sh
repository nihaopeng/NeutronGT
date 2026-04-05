#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# User-editable section
###############################################################################

# Recommended to update before running.
CUDA_VISIBLE_DEVICES_LIST="0,1,2,3"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
DATASET_DIR="${DATASET_DIR:-./dataset}"
LOG_DIR="${LOG_DIR:-./logs/efficiency_benchmark}"
RESULT_DIR="${RESULT_DIR:-./results/efficiency_benchmark}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# Optional:
# - Set to a positive integer to force one n_parts for every run.
# - Leave empty to use dataset+model specific defaults from get_n_parts().
DEFAULT_N_PARTS="${DEFAULT_N_PARTS:-}"

# Optional, simple run selection:
# - "all" (default) means use the full built-in list.
# - Or set comma-separated values, e.g.:
#     RUN_DATASETS="ogbn-arxiv,ogbn-products"
#     RUN_MODELS="GT,Graphormer-Slim"
RUN_DATASETS="${RUN_DATASETS:-all}"
RUN_MODELS="${RUN_MODELS:-all}"

# Required: fill these in before running.
GT_N_LAYERS="4"
GT_HIDDEN_DIM="128"
GT_NUM_HEADS="8"

GRAPHORMER_SLIM_N_LAYERS="4"
GRAPHORMER_SLIM_HIDDEN_DIM="64"
GRAPHORMER_SLIM_NUM_HEADS="8"

GRAPHORMER_LARGE_N_LAYERS="12"
GRAPHORMER_LARGE_HIDDEN_DIM="768"
GRAPHORMER_LARGE_NUM_HEADS="32"

###############################################################################
# Internal defaults
###############################################################################

EPOCHS=20
SEQ_LEN=6400
STRUCT_ENC=True
MAX_DIST=5
USE_CACHE=1
PPR_BACKEND=appnp
PPR_TOPK=5
PPR_ALPHA=0.85
PPR_NUM_ITERATIONS=10
PPR_BATCH_SIZE=8192
PPR_ITER_TOPK=5
DISTRIBUTED_TIMEOUT_MINUTES=120

DATASETS=(
  "ogbn-arxiv"
  "ogbn-products"
  "reddit"
  "AmazonProducts"
  "ogbn-papers100M"
)

MODEL_ALIASES=(
  "GT"
  "Graphormer-Slim"
  "Graphormer-Large"
)

DEVICE_COUNT=$(awk -F',' 'NF { print NF }' <<< "${CUDA_VISIBLE_DEVICES_LIST}")
RESULT_CSV="${RESULT_DIR}/efficiency_summary_${RUN_TAG}.csv"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_filled() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" || "${value}" == "__FILL_ME__" ]]; then
    fail "${name} is required. Please edit ${0} and fill it in before running."
  fi
}

require_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    fail "${name} must be an integer, got: ${value}"
  fi
}

csv_escape() {
  local value="${1:-}"
  value=${value//\"/\"\"}
  printf '"%s"' "${value}"
}

append_csv_header() {
  if [[ -f "${RESULT_CSV}" ]]; then
    return
  fi
  cat > "${RESULT_CSV}" <<'CSVEOF'
run_tag,dataset,model_alias,model_cli,attn_type,n_layers,hidden_dim,ffn_dim,num_heads,n_parts,status,graph_preprocess_total_time_s,window_state_distribution_total_time_s,window_bundle_write_time_s,window_rebuild_load_time_s,window_rebuild_dup_time_s,window_rebuild_subgraph_time_s,window_rebuild_spatial_time_s,window_rebuild_total_time_s,last_epoch,last_loss,last_train_time_s,last_cpu_to_gpu_time_s,last_window_fw_bw_avg_time_s,last_window_fw_bw_total_time_s,last_num_windows,last_train_acc,last_valid_acc,last_test_acc,log_path
CSVEOF
}

get_model_cli_name() {
  case "$1" in
    "GT") echo "gt" ;;
    "Graphormer-Slim"|"Graphormer-Large") echo "graphormer" ;;
    *) fail "Unknown model alias: $1" ;;
  esac
}

get_attn_type() {
  case "$1" in
    "GT") echo "sparse" ;;
    "Graphormer-Slim") echo "sparse" ;;
    "Graphormer-Large") echo "full" ;;
    *) fail "Unknown model alias: $1" ;;
  esac
}

get_n_layers() {
  case "$1" in
    "GT") echo "${GT_N_LAYERS}" ;;
    "Graphormer-Slim") echo "${GRAPHORMER_SLIM_N_LAYERS}" ;;
    "Graphormer-Large") echo "${GRAPHORMER_LARGE_N_LAYERS}" ;;
    *) fail "Unknown model alias: $1" ;;
  esac
}

get_hidden_dim() {
  case "$1" in
    "GT") echo "${GT_HIDDEN_DIM}" ;;
    "Graphormer-Slim") echo "${GRAPHORMER_SLIM_HIDDEN_DIM}" ;;
    "Graphormer-Large") echo "${GRAPHORMER_LARGE_HIDDEN_DIM}" ;;
    *) fail "Unknown model alias: $1" ;;
  esac
}

get_num_heads() {
  case "$1" in
    "GT") echo "${GT_NUM_HEADS}" ;;
    "Graphormer-Slim") echo "${GRAPHORMER_SLIM_NUM_HEADS}" ;;
    "Graphormer-Large") echo "${GRAPHORMER_LARGE_NUM_HEADS}" ;;
    *) fail "Unknown model alias: $1" ;;
  esac
}

contains_exact() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
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
  if [[ "${RUN_DATASETS}" == "all" ]]; then
    printf '%s\n' "${DATASETS[@]}"
    return
  fi
  local -a selected=()
  split_csv_to_array "${RUN_DATASETS}" selected
  printf '%s\n' "${selected[@]}"
}

get_selected_models() {
  if [[ "${RUN_MODELS}" == "all" ]]; then
    printf '%s\n' "${MODEL_ALIASES[@]}"
    return
  fi
  local -a selected=()
  split_csv_to_array "${RUN_MODELS}" selected
  printf '%s\n' "${selected[@]}"
}

get_n_parts() {
  local dataset="$1"
  local model_alias="$2"

  if [[ -n "${DEFAULT_N_PARTS}" ]]; then
    echo "${DEFAULT_N_PARTS}"
    return
  fi

  case "${dataset}:${model_alias}" in
    "ogbn-arxiv:GT") echo "32" ;;
    "ogbn-arxiv:Graphormer-Slim") echo "48" ;;
    "ogbn-arxiv:Graphormer-Large") echo "64" ;;

    "reddit:GT") echo "48" ;;
    "reddit:Graphormer-Slim") echo "64" ;;
    "reddit:Graphormer-Large") echo "96" ;;

    "AmazonProducts:GT") echo "96" ;;
    "AmazonProducts:Graphormer-Slim") echo "128" ;;
    "AmazonProducts:Graphormer-Large") echo "160" ;;

    "ogbn-products:GT") echo "128" ;;
    "ogbn-products:Graphormer-Slim") echo "160" ;;
    "ogbn-products:Graphormer-Large") echo "200" ;;

    "ogbn-papers100M:GT") echo "160" ;;
    "ogbn-papers100M:Graphormer-Slim") echo "200" ;;
    "ogbn-papers100M:Graphormer-Large") echo "256" ;;

    *) echo "50" ;;
  esac
}

extract_first_metric() {
  local log_file="$1"
  local prefix="$2"
  awk -F': ' -v prefix="${prefix}" '
    index($0, prefix) {
      value = $2
      sub(/s.*/, "", value)
      print value
      exit
    }
  ' "${log_file}"
}

extract_last_train_line() {
  local log_file="$1"
  grep "Train Time:" "${log_file}" | tail -n 1 || true
}

extract_last_eval_line() {
  local log_file="$1"
  grep "train_acc:" "${log_file}" | tail -n 1 || true
}

extract_rebuild_line() {
  local log_file="$1"
  grep -F "[rank 0] Window rebuild:" "${log_file}" | tail -n 1 || true
}

parse_value_by_regex() {
  local line="$1"
  local regex="$2"
  sed -nE "s/${regex}/\\1/p" <<< "${line}"
}

append_csv_row() {
  local row="$1"
  printf '%s\n' "${row}" >> "${RESULT_CSV}"
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
  local log_file="${10}"
  local row
  row="$(
    printf '%s,' "$(csv_escape "${RUN_TAG}")"
    printf '%s,' "$(csv_escape "${dataset}")"
    printf '%s,' "$(csv_escape "${model_alias}")"
    printf '%s,' "$(csv_escape "${model_cli}")"
    printf '%s,' "$(csv_escape "${attn_type}")"
    printf '%s,' "$(csv_escape "${n_layers}")"
    printf '%s,' "$(csv_escape "${hidden_dim}")"
    printf '%s,' "$(csv_escape "${ffn_dim}")"
    printf '%s,' "$(csv_escape "${num_heads}")"
    printf '%s,' "$(csv_escape "${n_parts}")"
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
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s,' "$(csv_escape "")"
    printf '%s' "$(csv_escape "${log_file}")"
  )"
  append_csv_row "${row}"
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
  local log_file="${10}"

  local graph_preprocess_total_time
  local window_state_distribution_total_time
  local window_bundle_write_time
  local rebuild_line
  local last_train_line
  local last_eval_line
  local rebuild_load_time
  local rebuild_dup_time
  local rebuild_subgraph_time
  local rebuild_spatial_time
  local rebuild_total_time
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

  graph_preprocess_total_time="$(extract_first_metric "${log_file}" "Graph preprocess total time:")"
  window_state_distribution_total_time="$(extract_first_metric "${log_file}" "Window state distribution total time:")"
  window_bundle_write_time="$(extract_first_metric "${log_file}" "Window bundle write time:")"
  rebuild_line="$(extract_rebuild_line "${log_file}")"
  last_train_line="$(extract_last_train_line "${log_file}")"
  last_eval_line="$(extract_last_eval_line "${log_file}")"

  rebuild_load_time="$(parse_value_by_regex "${rebuild_line}" '.*load=([0-9.]+)s, dup=.*')"
  rebuild_dup_time="$(parse_value_by_regex "${rebuild_line}" '.*dup=([0-9.]+)s, subgraph=.*')"
  rebuild_subgraph_time="$(parse_value_by_regex "${rebuild_line}" '.*subgraph=([0-9.]+)s, spatial=.*')"
  rebuild_spatial_time="$(parse_value_by_regex "${rebuild_line}" '.*spatial=([0-9.]+)s, total=.*')"
  rebuild_total_time="$(parse_value_by_regex "${rebuild_line}" '.*total=([0-9.]+)s.*')"

  last_epoch="$(parse_value_by_regex "${last_train_line}" '.*Epoch: ([0-9]+), Loss: .*')"
  last_loss="$(parse_value_by_regex "${last_train_line}" '.*Loss: ([0-9.]+), Train Time: .*')"
  last_train_time="$(parse_value_by_regex "${last_train_line}" '.*Train Time: ([0-9.]+)s, CPU->GPU Time: .*')"
  last_cpu_to_gpu_time="$(parse_value_by_regex "${last_train_line}" '.*CPU->GPU Time: ([0-9.]+)s, Window FW/BW Avg: .*')"
  last_window_fw_bw_avg_time="$(parse_value_by_regex "${last_train_line}" '.*Window FW/BW Avg: ([0-9.]+)s, Window FW/BW Total: .*')"
  last_window_fw_bw_total_time="$(parse_value_by_regex "${last_train_line}" '.*Window FW/BW Total: ([0-9.]+)s, Windows: .*')"
  last_num_windows="$(parse_value_by_regex "${last_train_line}" '.*Windows: ([0-9]+), Max Window Steps: .*')"

  last_train_acc="$(parse_value_by_regex "${last_eval_line}" '.*train_acc: ([0-9.]+)%, valid_acc: .*')"
  last_valid_acc="$(parse_value_by_regex "${last_eval_line}" '.*valid_acc: ([0-9.]+)%, test_acc: .*')"
  last_test_acc="$(parse_value_by_regex "${last_eval_line}" '.*test_acc: ([0-9.]+)%, samples.*')"

  row="$(
    printf '%s,' "$(csv_escape "${RUN_TAG}")"
    printf '%s,' "$(csv_escape "${dataset}")"
    printf '%s,' "$(csv_escape "${model_alias}")"
    printf '%s,' "$(csv_escape "${model_cli}")"
    printf '%s,' "$(csv_escape "${attn_type}")"
    printf '%s,' "$(csv_escape "${n_layers}")"
    printf '%s,' "$(csv_escape "${hidden_dim}")"
    printf '%s,' "$(csv_escape "${ffn_dim}")"
    printf '%s,' "$(csv_escape "${num_heads}")"
    printf '%s,' "$(csv_escape "${n_parts}")"
    printf '%s,' "$(csv_escape "SUCCESS")"
    printf '%s,' "$(csv_escape "${graph_preprocess_total_time}")"
    printf '%s,' "$(csv_escape "${window_state_distribution_total_time}")"
    printf '%s,' "$(csv_escape "${window_bundle_write_time}")"
    printf '%s,' "$(csv_escape "${rebuild_load_time}")"
    printf '%s,' "$(csv_escape "${rebuild_dup_time}")"
    printf '%s,' "$(csv_escape "${rebuild_subgraph_time}")"
    printf '%s,' "$(csv_escape "${rebuild_spatial_time}")"
    printf '%s,' "$(csv_escape "${rebuild_total_time}")"
    printf '%s,' "$(csv_escape "${last_epoch}")"
    printf '%s,' "$(csv_escape "${last_loss}")"
    printf '%s,' "$(csv_escape "${last_train_time}")"
    printf '%s,' "$(csv_escape "${last_cpu_to_gpu_time}")"
    printf '%s,' "$(csv_escape "${last_window_fw_bw_avg_time}")"
    printf '%s,' "$(csv_escape "${last_window_fw_bw_total_time}")"
    printf '%s,' "$(csv_escape "${last_num_windows}")"
    printf '%s,' "$(csv_escape "${last_train_acc}")"
    printf '%s,' "$(csv_escape "${last_valid_acc}")"
    printf '%s,' "$(csv_escape "${last_test_acc}")"
    printf '%s' "$(csv_escape "${log_file}")"
  )"
  append_csv_row "${row}"
}

validate_user_inputs() {
  require_filled "GT_N_LAYERS" "${GT_N_LAYERS}"
  require_filled "GT_HIDDEN_DIM" "${GT_HIDDEN_DIM}"
  require_filled "GT_NUM_HEADS" "${GT_NUM_HEADS}"
  require_filled "GRAPHORMER_SLIM_N_LAYERS" "${GRAPHORMER_SLIM_N_LAYERS}"
  require_filled "GRAPHORMER_SLIM_HIDDEN_DIM" "${GRAPHORMER_SLIM_HIDDEN_DIM}"
  require_filled "GRAPHORMER_SLIM_NUM_HEADS" "${GRAPHORMER_SLIM_NUM_HEADS}"
  require_filled "GRAPHORMER_LARGE_N_LAYERS" "${GRAPHORMER_LARGE_N_LAYERS}"
  require_filled "GRAPHORMER_LARGE_HIDDEN_DIM" "${GRAPHORMER_LARGE_HIDDEN_DIM}"
  require_filled "GRAPHORMER_LARGE_NUM_HEADS" "${GRAPHORMER_LARGE_NUM_HEADS}"

  require_integer "MASTER_PORT_BASE" "${MASTER_PORT_BASE}"
  require_integer "GT_N_LAYERS" "${GT_N_LAYERS}"
  require_integer "GT_HIDDEN_DIM" "${GT_HIDDEN_DIM}"
  require_integer "GT_NUM_HEADS" "${GT_NUM_HEADS}"
  require_integer "GRAPHORMER_SLIM_N_LAYERS" "${GRAPHORMER_SLIM_N_LAYERS}"
  require_integer "GRAPHORMER_SLIM_HIDDEN_DIM" "${GRAPHORMER_SLIM_HIDDEN_DIM}"
  require_integer "GRAPHORMER_SLIM_NUM_HEADS" "${GRAPHORMER_SLIM_NUM_HEADS}"
  require_integer "GRAPHORMER_LARGE_N_LAYERS" "${GRAPHORMER_LARGE_N_LAYERS}"
  require_integer "GRAPHORMER_LARGE_HIDDEN_DIM" "${GRAPHORMER_LARGE_HIDDEN_DIM}"
  require_integer "GRAPHORMER_LARGE_NUM_HEADS" "${GRAPHORMER_LARGE_NUM_HEADS}"

  if [[ "${DEVICE_COUNT}" -lt 1 ]]; then
    fail "CUDA_VISIBLE_DEVICES_LIST must contain at least one device."
  fi

  if [[ -n "${DEFAULT_N_PARTS}" ]]; then
    require_integer "DEFAULT_N_PARTS" "${DEFAULT_N_PARTS}"
  fi

  local -a selected_datasets=()
  local -a selected_models=()
  mapfile -t selected_datasets < <(get_selected_datasets)
  mapfile -t selected_models < <(get_selected_models)

  local dataset
  for dataset in "${selected_datasets[@]}"; do
    contains_exact "${dataset}" "${DATASETS[@]}" || fail "Unknown dataset in RUN_DATASETS: ${dataset}"
  done

  local model_alias
  for model_alias in "${selected_models[@]}"; do
    contains_exact "${model_alias}" "${MODEL_ALIASES[@]}" || fail "Unknown model in RUN_MODELS: ${model_alias}"
  done
}

run_one_experiment() {
  local dataset="$1"
  local model_alias="$2"
  local port="$3"
  local n_parts="$4"
  local model_cli
  local attn_type
  local n_layers
  local hidden_dim
  local ffn_dim
  local num_heads
  local log_file
  local command_status
  local -a cmd

  model_cli="$(get_model_cli_name "${model_alias}")"
  attn_type="$(get_attn_type "${model_alias}")"
  n_layers="$(get_n_layers "${model_alias}")"
  hidden_dim="$(get_hidden_dim "${model_alias}")"
  num_heads="$(get_num_heads "${model_alias}")"
  ffn_dim="${hidden_dim}"

  log_file="${LOG_DIR}/${dataset}__${model_alias}__${RUN_TAG}.log"

  cmd=(
    torchrun
    --nproc_per_node="${DEVICE_COUNT}"
    --master_port "${port}"
    main_sp_node_level_ppr.py
    --dataset "${dataset}"
    --dataset_dir "${DATASET_DIR}"
    --seq_len "${SEQ_LEN}"
    --n_layers "${n_layers}"
    --hidden_dim "${hidden_dim}"
    --ffn_dim "${ffn_dim}"
    --num_heads "${num_heads}"
    --epochs "${EPOCHS}"
    --model "${model_cli}"
    --distributed-backend "nccl"
    --attn_type "${attn_type}"
    --struct_enc "${STRUCT_ENC}"
    --max_dist "${MAX_DIST}"
    --use_cache "${USE_CACHE}"
    --n_parts "${n_parts}"
    --ppr_backend "${PPR_BACKEND}"
    --ppr_topk "${PPR_TOPK}"
    --ppr_alpha "${PPR_ALPHA}"
    --ppr_num_iterations "${PPR_NUM_ITERATIONS}"
    --ppr_batch_size "${PPR_BATCH_SIZE}"
    --ppr_iter_topk "${PPR_ITER_TOPK}"
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}"
  )

  echo "===================================================================================="
  echo "[RUN] dataset=${dataset} model=${model_alias} n_parts=${n_parts} port=${port}"
  echo "[RUN] log=${log_file}"
  echo "===================================================================================="

  set +e
  {
    printf '# run_tag=%s\n' "${RUN_TAG}"
    printf '# dataset=%s\n' "${dataset}"
    printf '# model_alias=%s\n' "${model_alias}"
    printf '# command='
    printf '%q ' env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}" "${cmd[@]}"
    printf '\n'
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}" "${cmd[@]}"
  } 2>&1 | tee "${log_file}"
  command_status=${PIPESTATUS[0]}
  set -e

  if [[ "${command_status}" -ne 0 ]]; then
    echo "[WARN] Experiment failed: dataset=${dataset}, model=${model_alias}, status=${command_status}" >&2
    append_failed_row "${dataset}" "${model_alias}" "${model_cli}" "${attn_type}" "${n_layers}" "${hidden_dim}" "${ffn_dim}" "${num_heads}" "${n_parts}" "${log_file}"
    return 0
  fi

  append_success_row "${dataset}" "${model_alias}" "${model_cli}" "${attn_type}" "${n_layers}" "${hidden_dim}" "${ffn_dim}" "${num_heads}" "${n_parts}" "${log_file}"
}

main() {
  local dataset
  local model_alias
  local current_port
  local n_parts
  local -a selected_datasets=()
  local -a selected_models=()

  validate_user_inputs

  mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
  append_csv_header

  mapfile -t selected_datasets < <(get_selected_datasets)
  mapfile -t selected_models < <(get_selected_models)

  echo "===================================================================================="
  echo "[CONFIG] datasets: ${selected_datasets[*]}"
  echo "[CONFIG] models: ${selected_models[*]}"
  if [[ -n "${DEFAULT_N_PARTS}" ]]; then
    echo "[CONFIG] n_parts: fixed=${DEFAULT_N_PARTS}"
  else
    echo "[CONFIG] n_parts: auto(dataset, model)"
  fi
  echo "===================================================================================="

  current_port="${MASTER_PORT_BASE}"
  for dataset in "${selected_datasets[@]}"; do
    for model_alias in "${selected_models[@]}"; do
      n_parts="$(get_n_parts "${dataset}" "${model_alias}")"
      run_one_experiment "${dataset}" "${model_alias}" "${current_port}" "${n_parts}"
      current_port=$((current_port + 1))
    done
  done

  echo "===================================================================================="
  echo "[DONE] Benchmark finished."
  echo "[DONE] CSV summary: ${RESULT_CSV}"
  echo "===================================================================================="
}

main "$@"
