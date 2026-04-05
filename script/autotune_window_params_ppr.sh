#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES_LIST=${CUDA_VISIBLE_DEVICES_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_LIST"
RUN_DATASETS=${RUN_DATASETS:-all}
RUN_MODELS=${RUN_MODELS:-all}
RESULT_DIR=${AUTOTUNE_RESULT_DIR:-"$ROOT_DIR/results/autotune_window_params_ppr"}
LOG_DIR="$RESULT_DIR/logs"
DUP_RATIO_MIN=${DUP_RATIO_MIN:-0.10}
DUP_RATIO_MAX=${DUP_RATIO_MAX:-0.20}
DUP_RATIO_IDEAL=${DUP_RATIO_IDEAL:-0.15}
GPU_MEM_BUDGET_RATIO=${GPU_MEM_BUDGET_RATIO:-0.85}
TUNE_EPOCHS=${TUNE_EPOCHS:-1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29650}
DATASET_DIR=${DATASET_DIR:-./dataset}
PYTHON_BIN=${PYTHON_BIN:-python}
SEQ_LEN=${SEQ_LEN:-6400}
STRUCT_ENC=${STRUCT_ENC:-True}
MAX_DIST=${MAX_DIST:-5}
USE_CACHE=${USE_CACHE:-1}
PPR_BACKEND=${PPR_BACKEND:-appnp}
PPR_TOPK=${PPR_TOPK:-5}
PPR_ALPHA=${PPR_ALPHA:-0.85}
PPR_NUM_ITERATIONS=${PPR_NUM_ITERATIONS:-10}
PPR_BATCH_SIZE=${PPR_BATCH_SIZE:-8192}
PPR_ITER_TOPK=${PPR_ITER_TOPK:-5}
DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-120}
DROPOUT_RATE=${DROPOUT_RATE:-0.0}
ATTN_DROPOUT_RATE=${ATTN_DROPOUT_RATE:-0.0}
INPUT_DROPOUT_RATE=${INPUT_DROPOUT_RATE:-0.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
PEAK_LR=${PEAK_LR:-0.001}
END_LR=${END_LR:-1e-9}
WARMUP_UPDATES=${WARMUP_UPDATES:-0}
PATIENCE=${PATIENCE:-50}
NUM_GLOBAL_NODE=${NUM_GLOBAL_NODE:-1}

GT_N_LAYERS=${GT_N_LAYERS:-4}
GT_HIDDEN_DIM=${GT_HIDDEN_DIM:-128}
GT_NUM_HEADS=${GT_NUM_HEADS:-8}
GRAPHORMER_SLIM_N_LAYERS=${GRAPHORMER_SLIM_N_LAYERS:-4}
GRAPHORMER_SLIM_HIDDEN_DIM=${GRAPHORMER_SLIM_HIDDEN_DIM:-64}
GRAPHORMER_SLIM_NUM_HEADS=${GRAPHORMER_SLIM_NUM_HEADS:-8}
GRAPHORMER_LARGE_N_LAYERS=${GRAPHORMER_LARGE_N_LAYERS:-12}
GRAPHORMER_LARGE_HIDDEN_DIM=${GRAPHORMER_LARGE_HIDDEN_DIM:-768}
GRAPHORMER_LARGE_NUM_HEADS=${GRAPHORMER_LARGE_NUM_HEADS:-32}

IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES_LIST"
GPU_COUNT=${#VISIBLE_GPU_ARRAY[@]}
[[ "$DATASET_DIR" != */ ]] && DATASET_DIR="${DATASET_DIR}/"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

CSV_HEADER='dataset,model_alias,attn_type,n_parts,related_nodes_topk_rate,gpu_total_mem_mb,gpu_budget_mb,dynamic_target_max_window_nodes,max_window_nodes,avg_window_nodes,avg_dup_nodes_per_window,avg_dup_ratio_per_window,peak_gpu_mem_mb,train_epoch_time,status,log_path'

IFS=',' read -r -a RUN_DATASET_FILTER <<< "$RUN_DATASETS"
IFS=',' read -r -a RUN_MODEL_FILTER <<< "$RUN_MODELS"

DATASETS=("ogbn-arxiv" "reddit" "AmazonProducts" "ogbn-products")
MODELS=("GT" "Graphormer-Slim" "Graphormer-Large")

contains_item() {
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

should_run_dataset() {
    local dataset="$1"
    if [[ "$RUN_DATASETS" == "all" ]]; then
        return 0
    fi
    contains_item "$dataset" "${RUN_DATASET_FILTER[@]}"
}

should_run_model() {
    local model="$1"
    if [[ "$RUN_MODELS" == "all" ]]; then
        return 0
    fi
    contains_item "$model" "${RUN_MODEL_FILTER[@]}"
}

ensure_dataset_csvs() {
    local dataset="$1"
    local safe_dataset="${dataset//\//_}"
    local trials_csv="$RESULT_DIR/trials_${safe_dataset}.csv"
    local summary_csv="$RESULT_DIR/summary_${safe_dataset}.csv"
    if [[ ! -f "$trials_csv" ]]; then
        printf '%s\n' "$CSV_HEADER" > "$trials_csv"
    fi
    if [[ ! -f "$summary_csv" ]]; then
        printf '%s\n' "$CSV_HEADER" > "$summary_csv"
    fi
}

get_trials_csv_path() {
    local dataset="$1"
    local safe_dataset="${dataset//\//_}"
    printf '%s/trials_%s.csv' "$RESULT_DIR" "$safe_dataset"
}

get_summary_csv_path() {
    local dataset="$1"
    local safe_dataset="${dataset//\//_}"
    printf '%s/summary_%s.csv' "$RESULT_DIR" "$safe_dataset"
}

get_model_args() {
    local model_alias="$1"
    case "$model_alias" in
        GT)
            printf '%s|%s|%s|%s|%s|%s' "gt" "sparse" "$GT_N_LAYERS" "$GT_HIDDEN_DIM" "$GT_NUM_HEADS" "$GT_HIDDEN_DIM"
            ;;
        Graphormer-Slim)
            printf '%s|%s|%s|%s|%s|%s' "graphormer" "sparse" "$GRAPHORMER_SLIM_N_LAYERS" "$GRAPHORMER_SLIM_HIDDEN_DIM" "$GRAPHORMER_SLIM_NUM_HEADS" "$GRAPHORMER_SLIM_HIDDEN_DIM"
            ;;
        Graphormer-Large)
            printf '%s|%s|%s|%s|%s|%s' "graphormer" "sparse" "$GRAPHORMER_LARGE_N_LAYERS" "$GRAPHORMER_LARGE_HIDDEN_DIM" "$GRAPHORMER_LARGE_NUM_HEADS" "$GRAPHORMER_LARGE_HIDDEN_DIM"
            ;;
        *)
            return 1
            ;;
    esac
}

get_search_config() {
    local dataset="$1"
    local model_alias="$2"
    case "$dataset:$model_alias" in
        ogbn-arxiv:GT) printf '24 8 120' ;;
        ogbn-arxiv:Graphormer-Slim) printf '32 8 128' ;;
        ogbn-arxiv:Graphormer-Large) printf '96 16 256' ;;
        reddit:GT) printf '40 8 160' ;;
        reddit:Graphormer-Slim) printf '56 8 192' ;;
        reddit:Graphormer-Large) printf '128 16 320' ;;
        AmazonProducts:GT) printf '72 16 224' ;;
        AmazonProducts:Graphormer-Slim) printf '96 16 256' ;;
        AmazonProducts:Graphormer-Large) printf '160 32 416' ;;
        ogbn-products:GT) printf '96 16 288' ;;
        ogbn-products:Graphormer-Slim) printf '128 16 352' ;;
        ogbn-products:Graphormer-Large) printf '192 32 512' ;;
        *) return 1 ;;
    esac
}

trim_whitespace() {
    local value="$1"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    printf '%s' "$value"
}

detect_gpu_total_mem_mb() {
    local query_output
    query_output=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -z "$query_output" ]]; then
        echo 24576
        return
    fi
    python - "$CUDA_VISIBLE_DEVICES_LIST" <<'PY'
import subprocess
import sys
visible = [item.strip() for item in sys.argv[1].split(',') if item.strip()]
out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits'], text=True)
by_index = {}
for line in out.strip().splitlines():
    idx, mem = [part.strip() for part in line.split(',')]
    by_index[idx] = int(float(mem))
vals = [by_index[idx] for idx in visible if idx in by_index]
print(min(vals) if vals else 24576)
PY
}

compute_dynamic_target_max_window_nodes() {
    local gpu_total_mem_mb="$1"
    local n_layers="$2"
    local hidden_dim="$3"
    python - "$gpu_total_mem_mb" "$n_layers" "$hidden_dim" <<'PY'
import math
import sys
gpu_total_mem_mb = float(sys.argv[1])
n_layers = float(sys.argv[2])
hidden_dim = float(sys.argv[3])
ref_cost = 4.0 * 64.0
model_cost = n_layers * hidden_dim
estimated = math.floor(9500.0 * ((gpu_total_mem_mb / 1024.0) / 24.0) * math.sqrt(ref_cost / model_cost))
estimated = min(9900, estimated)
estimated = max(2500, estimated)
print(int(estimated))
PY
}

compute_gpu_budget_mb() {
    local gpu_total_mem_mb="$1"
    python - "$gpu_total_mem_mb" "$GPU_MEM_BUDGET_RATIO" <<'PY'
import math
import sys
print(int(math.floor(float(sys.argv[1]) * float(sys.argv[2]))))
PY
}

parse_window_stats() {
    local log_path="$1"
    local line
    line=$(grep 'Window stats:' "$log_path" | tail -n 1 || true)
    if [[ -z "$line" ]]; then
        return 1
    fi
    python - "$line" <<'PY'
import re
import sys
line = sys.argv[1]
pattern = re.compile(
    r'max_window_nodes=(?P<max>[0-9]+),\s*'
    r'avg_window_nodes=(?P<avg>[0-9]+(?:\.[0-9]+)?),\s*'
    r'avg_dup_nodes_per_window=(?P<dup_nodes>[0-9]+(?:\.[0-9]+)?),\s*'
    r'avg_dup_ratio_per_window=(?P<dup_ratio>[0-9]+(?:\.[0-9]+)?)'
)
match = pattern.search(line)
if not match:
    sys.exit(1)
print(match.group('max'))
print(match.group('avg'))
print(match.group('dup_nodes'))
print(match.group('dup_ratio'))
PY
}

parse_train_epoch_time() {
    local log_path="$1"
    local line
    line=$(grep 'Training epoch time:' "$log_path" | tail -n 1 || true)
    if [[ -z "$line" ]]; then
        printf ''
        return 0
    fi
    python - "$line" <<'PY'
import re
import sys
line = sys.argv[1]
match = re.search(r'Training epoch time:\s*([0-9]+(?:\.[0-9]+)?)s', line)
if not match:
    sys.exit(1)
print(match.group(1))
PY
}

parse_peak_gpu_mem_mb() {
    local log_path="$1"
    local line
    line=$(grep 'Peak GPU memory summary:' "$log_path" | tail -n 1 || true)
    if [[ -z "$line" ]]; then
        printf ''
        return 0
    fi
    python - "$line" <<'PY'
import re
import sys
line = sys.argv[1]
match = re.search(r'max_reserved_mb=([0-9]+(?:\.[0-9]+)?),\s*max_allocated_mb=([0-9]+(?:\.[0-9]+)?)', line)
if not match:
    sys.exit(1)
print(match.group(1))
PY
}

append_csv_row() {
    local csv_path="$1"
    shift
    printf '%s\n' "$*" >> "$csv_path"
}

run_case() {
    local mode="$1"
    local dataset="$2"
    local model_alias="$3"
    local n_parts="$4"
    local related_topk="$5"
    local port="$6"
    local log_path="$7"

    local model_name attn_type n_layers hidden_dim num_heads ffn_dim
    IFS='|' read -r model_name attn_type n_layers hidden_dim num_heads ffn_dim <<< "$(get_model_args "$model_alias")"

    local epochs="$TUNE_EPOCHS"
    local preprocess_only=0
    if [[ "$mode" == "preprocess" ]]; then
        epochs=0
        preprocess_only=1
    fi

    local cmd=(
        torchrun
        --nproc_per_node="$GPU_COUNT"
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
        --model "$model_name"
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
        --preprocess_only "$preprocess_only"
    )

    printf '[AUTOTUNE] mode=%s dataset=%s model=%s n_parts=%s related_topk=%s log=%s\n' "$mode" "$dataset" "$model_alias" "$n_parts" "$related_topk" "$log_path"
    "${cmd[@]}" > "$log_path" 2>&1
    return $?
}

choose_better_dup_match() {
    local current_ratio="$1"
    local best_ratio="$2"
    python - "$current_ratio" "$best_ratio" "$DUP_RATIO_IDEAL" <<'PY'
import sys
current = float(sys.argv[1])
best = float(sys.argv[2])
ideal = float(sys.argv[3])
print('1' if abs(current - ideal) < abs(best - ideal) else '0')
PY
}

is_dup_ratio_in_range() {
    local ratio="$1"
    python - "$ratio" "$DUP_RATIO_MIN" "$DUP_RATIO_MAX" <<'PY'
import sys
value = float(sys.argv[1])
lo = float(sys.argv[2])
hi = float(sys.argv[3])
print('1' if lo <= value <= hi else '0')
PY
}

is_within_budget() {
    local peak_mem_mb="$1"
    local budget_mb="$2"
    python - "$peak_mem_mb" "$budget_mb" <<'PY'
import sys
print('1' if float(sys.argv[1]) <= float(sys.argv[2]) else '0')
PY
}

log_contains_oom() {
    local log_path="$1"
    if grep -q -E 'CUDA out of memory|torch.cuda.OutOfMemoryError|OOM' "$log_path"; then
        return 0
    fi
    return 1
}

case_index=0
gpu_total_mem_mb=$(detect_gpu_total_mem_mb)
gpu_budget_mb=$(compute_gpu_budget_mb "$gpu_total_mem_mb")

for dataset in "${DATASETS[@]}"; do
    if ! should_run_dataset "$dataset"; then
        continue
    fi
    ensure_dataset_csvs "$dataset"
    TRIALS_CSV=$(get_trials_csv_path "$dataset")
    SUMMARY_CSV=$(get_summary_csv_path "$dataset")

    for model_alias in "${MODELS[@]}"; do
        if ! should_run_model "$model_alias"; then
            continue
        fi

        IFS='|' read -r model_name attn_type n_layers hidden_dim num_heads ffn_dim <<< "$(get_model_args "$model_alias")"
        dynamic_target_max_window_nodes=$(compute_dynamic_target_max_window_nodes "$gpu_total_mem_mb" "$n_layers" "$hidden_dim")
        read -r start_n_parts step_n_parts max_n_parts <<< "$(get_search_config "$dataset" "$model_alias")"

        final_status="NO_WINDOW_SIZE_MATCH"
        final_n_parts=""
        final_related_topk=""
        final_max_window=""
        final_avg_window=""
        final_avg_dup_nodes=""
        final_avg_dup_ratio=""
        final_peak_gpu_mem_mb=""
        final_train_epoch_time=""
        final_log_path=""
        best_dup_n_parts=""
        best_dup_related_topk=""
        best_dup_max_window=""
        best_dup_avg_window=""
        best_dup_avg_dup_nodes=""
        best_dup_ratio=""
        best_dup_log_path=""

        n_parts="$start_n_parts"
        while [[ "$n_parts" -le "$max_n_parts" ]]; do
            case_index=$((case_index + 1))
            stage1_log="$LOG_DIR/${dataset}_${model_alias}_n${n_parts}_r2_stage1.log"
            port=$((MASTER_PORT_BASE + case_index))
            if ! run_case preprocess "$dataset" "$model_alias" "$n_parts" 2 "$port" "$stage1_log"; then
                append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,2,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,,,,,,,COMMAND_FAILED,$stage1_log"
                n_parts=$((n_parts + step_n_parts))
                continue
            fi

            mapfile -t stage1_stats < <(parse_window_stats "$stage1_log" 2>/dev/null || true)
            if [[ "${#stage1_stats[@]}" -ne 4 ]]; then
                append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,2,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,,,,,,,COMMAND_FAILED,$stage1_log"
                n_parts=$((n_parts + step_n_parts))
                continue
            fi

            max_window_nodes="$(trim_whitespace "${stage1_stats[0]}")"
            avg_window_nodes="$(trim_whitespace "${stage1_stats[1]}")"
            avg_dup_nodes_per_window="$(trim_whitespace "${stage1_stats[2]}")"
            avg_dup_ratio_per_window="$(trim_whitespace "${stage1_stats[3]}")"
            append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,2,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$max_window_nodes,$avg_window_nodes,$avg_dup_nodes_per_window,$avg_dup_ratio_per_window,,,WINDOW_SCAN,$stage1_log"

            if [[ "$max_window_nodes" -gt "$dynamic_target_max_window_nodes" ]]; then
                n_parts=$((n_parts + step_n_parts))
                continue
            fi

            matched_related_topk=""
            matched_max_window=""
            matched_avg_window=""
            matched_avg_dup_nodes=""
            matched_avg_dup_ratio=""

            for related_topk in 1 2 3 4 5 6; do
                case_index=$((case_index + 1))
                log_path="$LOG_DIR/${dataset}_${model_alias}_n${n_parts}_r${related_topk}.log"
                port=$((MASTER_PORT_BASE + case_index))
                if ! run_case preprocess "$dataset" "$model_alias" "$n_parts" "$related_topk" "$port" "$log_path"; then
                    append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,,,,,,,COMMAND_FAILED,$log_path"
                    continue
                fi

                mapfile -t stats < <(parse_window_stats "$log_path" 2>/dev/null || true)
                if [[ "${#stats[@]}" -ne 4 ]]; then
                    append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,,,,,,,COMMAND_FAILED,$log_path"
                    continue
                fi

                max_window_nodes="$(trim_whitespace "${stats[0]}")"
                avg_window_nodes="$(trim_whitespace "${stats[1]}")"
                avg_dup_nodes_per_window="$(trim_whitespace "${stats[2]}")"
                avg_dup_ratio_per_window="$(trim_whitespace "${stats[3]}")"
                append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$max_window_nodes,$avg_window_nodes,$avg_dup_nodes_per_window,$avg_dup_ratio_per_window,,,EVALUATED,$log_path"

                if [[ "$max_window_nodes" -gt "$dynamic_target_max_window_nodes" ]]; then
                    continue
                fi

                in_range=$(is_dup_ratio_in_range "$avg_dup_ratio_per_window")
                if [[ "$in_range" == "1" ]]; then
                    if [[ -z "$matched_avg_dup_ratio" ]]; then
                        matched_related_topk="$related_topk"
                        matched_max_window="$max_window_nodes"
                        matched_avg_window="$avg_window_nodes"
                        matched_avg_dup_nodes="$avg_dup_nodes_per_window"
                        matched_avg_dup_ratio="$avg_dup_ratio_per_window"
                    else
                        better=$(choose_better_dup_match "$avg_dup_ratio_per_window" "$matched_avg_dup_ratio")
                        if [[ "$better" == "1" ]]; then
                            matched_related_topk="$related_topk"
                            matched_max_window="$max_window_nodes"
                            matched_avg_window="$avg_window_nodes"
                            matched_avg_dup_nodes="$avg_dup_nodes_per_window"
                            matched_avg_dup_ratio="$avg_dup_ratio_per_window"
                        fi
                    fi
                else
                    if [[ -z "$best_dup_ratio" ]]; then
                        best_dup_n_parts="$n_parts"
                        best_dup_related_topk="$related_topk"
                        best_dup_max_window="$max_window_nodes"
                        best_dup_avg_window="$avg_window_nodes"
                        best_dup_avg_dup_nodes="$avg_dup_nodes_per_window"
                        best_dup_ratio="$avg_dup_ratio_per_window"
                        best_dup_log_path="$log_path"
                    else
                        better=$(choose_better_dup_match "$avg_dup_ratio_per_window" "$best_dup_ratio")
                        if [[ "$better" == "1" ]]; then
                            best_dup_n_parts="$n_parts"
                            best_dup_related_topk="$related_topk"
                            best_dup_max_window="$max_window_nodes"
                            best_dup_avg_window="$avg_window_nodes"
                            best_dup_avg_dup_nodes="$avg_dup_nodes_per_window"
                            best_dup_ratio="$avg_dup_ratio_per_window"
                            best_dup_log_path="$log_path"
                        fi
                    fi
                fi
            done

            if [[ -z "$matched_related_topk" ]]; then
                n_parts=$((n_parts + step_n_parts))
                continue
            fi

            case_index=$((case_index + 1))
            train_log="$LOG_DIR/${dataset}_${model_alias}_n${n_parts}_r${matched_related_topk}_train.log"
            port=$((MASTER_PORT_BASE + case_index))
            if ! run_case train "$dataset" "$model_alias" "$n_parts" "$matched_related_topk" "$port" "$train_log"; then
                if log_contains_oom "$train_log"; then
                    append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$matched_related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$matched_max_window,$matched_avg_window,$matched_avg_dup_nodes,$matched_avg_dup_ratio,,,OOM_AT_TRAIN,$train_log"
                else
                    append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$matched_related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$matched_max_window,$matched_avg_window,$matched_avg_dup_nodes,$matched_avg_dup_ratio,,,COMMAND_FAILED,$train_log"
                fi
                n_parts=$((n_parts + step_n_parts))
                continue
            fi

            train_epoch_time="$(parse_train_epoch_time "$train_log")"
            peak_gpu_mem_mb="$(parse_peak_gpu_mem_mb "$train_log")"
            append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$matched_related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$matched_max_window,$matched_avg_window,$matched_avg_dup_nodes,$matched_avg_dup_ratio,$peak_gpu_mem_mb,$train_epoch_time,TRAIN_VALIDATED,$train_log"

            if [[ -z "$peak_gpu_mem_mb" ]]; then
                append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$matched_related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$matched_max_window,$matched_avg_window,$matched_avg_dup_nodes,$matched_avg_dup_ratio,$peak_gpu_mem_mb,$train_epoch_time,COMMAND_FAILED,$train_log"
                n_parts=$((n_parts + step_n_parts))
                continue
            fi

            within_budget=$(is_within_budget "$peak_gpu_mem_mb" "$gpu_budget_mb")
            if [[ "$within_budget" != "1" ]]; then
                append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$matched_related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$matched_max_window,$matched_avg_window,$matched_avg_dup_nodes,$matched_avg_dup_ratio,$peak_gpu_mem_mb,$train_epoch_time,OOM_AT_TRAIN,$train_log"
                n_parts=$((n_parts + step_n_parts))
                continue
            fi

            final_status="SUCCESS"
            final_n_parts="$n_parts"
            final_related_topk="$matched_related_topk"
            final_max_window="$matched_max_window"
            final_avg_window="$matched_avg_window"
            final_avg_dup_nodes="$matched_avg_dup_nodes"
            final_avg_dup_ratio="$matched_avg_dup_ratio"
            final_peak_gpu_mem_mb="$peak_gpu_mem_mb"
            final_train_epoch_time="$train_epoch_time"
            final_log_path="$train_log"
            break
        done

        if [[ "$final_status" != "SUCCESS" && -n "$best_dup_ratio" ]]; then
            final_status="NO_DUP_RATIO_MATCH"
            final_n_parts="$best_dup_n_parts"
            final_related_topk="$best_dup_related_topk"
            final_max_window="$best_dup_max_window"
            final_avg_window="$best_dup_avg_window"
            final_avg_dup_nodes="$best_dup_avg_dup_nodes"
            final_avg_dup_ratio="$best_dup_ratio"
            final_log_path="$best_dup_log_path"
        fi

        append_csv_row "$SUMMARY_CSV" "$dataset,$model_alias,$attn_type,$final_n_parts,$final_related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$final_max_window,$final_avg_window,$final_avg_dup_nodes,$final_avg_dup_ratio,$final_peak_gpu_mem_mb,$final_train_epoch_time,$final_status,$final_log_path"
        printf '[AUTOTUNE][SUMMARY] dataset=%s model=%s attn=%s target_max_window=%s budget_mb=%s status=%s n_parts=%s related_topk=%s peak_gpu_mem_mb=%s train_epoch_time=%s\n' \
            "$dataset" "$model_alias" "$attn_type" "$dynamic_target_max_window_nodes" "$gpu_budget_mb" "$final_status" "$final_n_parts" "$final_related_topk" "$final_peak_gpu_mem_mb" "$final_train_epoch_time"
    done
done

echo
echo 'Final recommended parameter tables by dataset:'
for dataset in "${DATASETS[@]}"; do
    if ! should_run_dataset "$dataset"; then
        continue
    fi
    echo "===== $dataset ====="
    cat "$(get_summary_csv_path "$dataset")"
    echo
done
