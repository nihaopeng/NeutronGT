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
DUP_RATIO_MIN=${DUP_RATIO_MIN:-0.15}
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
MAX_TRAIN_VALIDATIONS_PER_CASE=${MAX_TRAIN_VALIDATIONS_PER_CASE:-2}
WINDOW_TARGET_SOFT_WEIGHT=${WINDOW_TARGET_SOFT_WEIGHT:-6.0}
DUP_RATIO_SOFT_WEIGHT=${DUP_RATIO_SOFT_WEIGHT:-8.0}
NPARTS_SOFT_WEIGHT=${NPARTS_SOFT_WEIGHT:-1.0}

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

CSV_HEADER='dataset,model_alias,attn_type,n_parts,related_nodes_topk_rate,gpu_total_mem_mb,gpu_budget_mb,dynamic_target_max_window_nodes,max_window_nodes,avg_window_nodes,avg_dup_nodes_per_window,avg_dup_ratio_per_window,peak_gpu_mem_mb,train_epoch_time,candidate_score,search_phase,selection_reason,status,log_path'

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
            printf '%s|%s|%s|%s|%s|%s' "gt_sw" "sparse" "$GT_N_LAYERS" "$GT_HIDDEN_DIM" "$GT_NUM_HEADS" "$GT_HIDDEN_DIM"
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

compute_candidate_score() {
    local max_window_nodes="$1"
    local dynamic_target_max_window_nodes="$2"
    local avg_dup_ratio_per_window="$3"
    local n_parts="$4"
    local start_n_parts="$5"
    python - "$max_window_nodes" "$dynamic_target_max_window_nodes" "$avg_dup_ratio_per_window" "$n_parts" "$start_n_parts" "$DUP_RATIO_MIN" "$DUP_RATIO_MAX" "$WINDOW_TARGET_SOFT_WEIGHT" "$DUP_RATIO_SOFT_WEIGHT" "$NPARTS_SOFT_WEIGHT" <<'PY'
import sys
max_window = float(sys.argv[1])
target = float(sys.argv[2])
dup = float(sys.argv[3])
n_parts = float(sys.argv[4])
start_n_parts = max(float(sys.argv[5]), 1.0)
dup_lo = float(sys.argv[6])
dup_hi = float(sys.argv[7])
w_window = float(sys.argv[8])
w_dup = float(sys.argv[9])
w_nparts = float(sys.argv[10])
window_penalty = max(0.0, max_window - target) / max(target, 1.0)
if dup < dup_lo:
    dup_penalty = (dup_lo - dup) / max(dup_hi - dup_lo, 1e-6)
elif dup > dup_hi:
    dup_penalty = (dup - dup_hi) / max(dup_hi - dup_lo, 1e-6)
else:
    dup_penalty = 0.0
nparts_penalty = n_parts / start_n_parts
score = w_window * window_penalty + w_dup * dup_penalty + w_nparts * nparts_penalty
print(f"{score:.6f}")
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

is_less_than() {
    local lhs="$1"
    local rhs="$2"
    python - "$lhs" "$rhs" <<'PY'
import sys
print('1' if float(sys.argv[1]) < float(sys.argv[2]) else '0')
PY
}

log_contains_oom() {
    local log_path="$1"
    if grep -q -E 'CUDA out of memory|torch.cuda.OutOfMemoryError|OOM' "$log_path"; then
        return 0
    fi
    return 1
}

evaluate_preprocess_candidate() {
    local dataset="$1"
    local model_alias="$2"
    local attn_type="$3"
    local n_parts="$4"
    local related_topk="$5"
    local dynamic_target_max_window_nodes="$6"
    local start_n_parts="$7"
    local search_phase="$8"
    local selection_reason="$9"
    local log_path="${10}"

    case_index=$((case_index + 1))
    local port=$((MASTER_PORT_BASE + case_index))
    if ! run_case preprocess "$dataset" "$model_alias" "$n_parts" "$related_topk" "$port" "$log_path"; then
        append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,,,,,,,,$search_phase,$selection_reason,COMMAND_FAILED,$log_path"
        return 1
    fi

    mapfile -t stats < <(parse_window_stats "$log_path" 2>/dev/null || true)
    if [[ "${#stats[@]}" -ne 4 ]]; then
        append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,,,,,,,,$search_phase,$selection_reason,COMMAND_FAILED,$log_path"
        return 1
    fi

    PRE_MAX_WINDOW_NODES="$(trim_whitespace "${stats[0]}")"
    PRE_AVG_WINDOW_NODES="$(trim_whitespace "${stats[1]}")"
    PRE_AVG_DUP_NODES_PER_WINDOW="$(trim_whitespace "${stats[2]}")"
    PRE_AVG_DUP_RATIO_PER_WINDOW="$(trim_whitespace "${stats[3]}")"
    PRE_CANDIDATE_SCORE="$(compute_candidate_score "$PRE_MAX_WINDOW_NODES" "$dynamic_target_max_window_nodes" "$PRE_AVG_DUP_RATIO_PER_WINDOW" "$n_parts" "$start_n_parts")"
    append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$PRE_MAX_WINDOW_NODES,$PRE_AVG_WINDOW_NODES,$PRE_AVG_DUP_NODES_PER_WINDOW,$PRE_AVG_DUP_RATIO_PER_WINDOW,,,$PRE_CANDIDATE_SCORE,$search_phase,$selection_reason,EVALUATED,$log_path"
    return 0
}

append_preproc_candidate() {
    local candidate_file="$1"
    local n_parts="$2"
    local related_topk="$3"
    local max_window_nodes="$4"
    local avg_window_nodes="$5"
    local avg_dup_nodes_per_window="$6"
    local avg_dup_ratio_per_window="$7"
    local candidate_score="$8"
    local search_phase="$9"
    local selection_reason="${10}"
    local log_path="${11}"
    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "$candidate_score" "$n_parts" "$related_topk" "$max_window_nodes" "$avg_window_nodes" "$avg_dup_nodes_per_window" "$avg_dup_ratio_per_window" "$search_phase" "$selection_reason" "$log_path" >> "$candidate_file"
}

get_adaptive_r_sequence() {
    local coarse_file="$1"
    python - "$coarse_file" <<'PY'
import math
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
rows = []
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    score, r, ratio = line.split('|')
    rows.append((float(score), int(r), float(ratio)))
rows.sort()
seen = {r for _, r, _ in rows}
if not rows:
    print('2 4 6')
    sys.exit(0)
_, best_r, _ = rows[0]
extras = []
for delta in (1, -1, 2, -2, 3, -3):
    candidate = best_r + delta
    if 1 <= candidate <= 6 and candidate not in seen and candidate not in extras:
        extras.append(candidate)
    if len(extras) >= 3:
        break
print(' '.join(str(x) for x in extras))
PY
}

collect_boundary_n_candidates() {
    local stage1_file="$1"
    local preferred_upper="$2"
    local step_n_parts="$3"
    local max_n_parts="$4"
    python - "$stage1_file" "$preferred_upper" "$step_n_parts" "$max_n_parts" <<'PY'
import pathlib
import sys
stage1 = pathlib.Path(sys.argv[1])
preferred_upper = int(sys.argv[2]) if sys.argv[2] else None
step = max(int(sys.argv[3]), 1)
max_n = int(sys.argv[4])
vals = set()
if preferred_upper is not None:
    vals.add(preferred_upper)
    if preferred_upper - step > 0:
        vals.add(preferred_upper - step)
    if preferred_upper + step <= max_n:
        vals.add(preferred_upper + step)
else:
    rows = []
    for line in stage1.read_text().splitlines():
        if not line.strip():
            continue
        score, n = line.split('|')
        rows.append((float(score), int(n)))
    rows.sort()
    for _, n in rows[:3]:
        vals.add(n)
print(' '.join(str(x) for x in sorted(vals)))
PY
}

select_top_candidates() {
    local candidate_file="$1"
    local top_k="$2"
    sort -t '|' -k1,1g "$candidate_file" | awk -F'|' '!seen[$2":"$3]++' | head -n "$top_k"
}

validate_train_candidate() {
    local dataset="$1"
    local model_alias="$2"
    local attn_type="$3"
    local n_parts="$4"
    local related_topk="$5"
    local dynamic_target_max_window_nodes="$6"
    local max_window_nodes="$7"
    local avg_window_nodes="$8"
    local avg_dup_nodes_per_window="$9"
    local avg_dup_ratio_per_window="${10}"
    local candidate_score="${11}"
    local search_phase="${12}"
    local selection_reason="${13}"

    local train_log="$LOG_DIR/${dataset}_${model_alias}_n${n_parts}_r${related_topk}_train.log"
    case_index=$((case_index + 1))
    local port=$((MASTER_PORT_BASE + case_index))
    if ! run_case train "$dataset" "$model_alias" "$n_parts" "$related_topk" "$port" "$train_log"; then
        local status="COMMAND_FAILED"
        if log_contains_oom "$train_log"; then
            status="OOM_AT_TRAIN"
        fi
        append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$max_window_nodes,$avg_window_nodes,$avg_dup_nodes_per_window,$avg_dup_ratio_per_window,,,$candidate_score,$search_phase,$selection_reason,$status,$train_log"
        TRAIN_VALIDATE_STATUS="$status"
        return 1
    fi

    TRAIN_EPOCH_TIME="$(parse_train_epoch_time "$train_log")"
    TRAIN_PEAK_GPU_MEM_MB="$(parse_peak_gpu_mem_mb "$train_log")"
    if [[ -z "$TRAIN_PEAK_GPU_MEM_MB" ]]; then
        append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$max_window_nodes,$avg_window_nodes,$avg_dup_nodes_per_window,$avg_dup_ratio_per_window,$TRAIN_PEAK_GPU_MEM_MB,$TRAIN_EPOCH_TIME,$candidate_score,$search_phase,$selection_reason,COMMAND_FAILED,$train_log"
        TRAIN_VALIDATE_STATUS="COMMAND_FAILED"
        return 1
    fi

    local within_budget
    within_budget=$(is_within_budget "$TRAIN_PEAK_GPU_MEM_MB" "$gpu_budget_mb")
    if [[ "$within_budget" != "1" ]]; then
        append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$max_window_nodes,$avg_window_nodes,$avg_dup_nodes_per_window,$avg_dup_ratio_per_window,$TRAIN_PEAK_GPU_MEM_MB,$TRAIN_EPOCH_TIME,$candidate_score,$search_phase,$selection_reason,OOM_AT_TRAIN,$train_log"
        TRAIN_VALIDATE_STATUS="OOM_AT_TRAIN"
        return 1
    fi

    append_csv_row "$TRIALS_CSV" "$dataset,$model_alias,$attn_type,$n_parts,$related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$max_window_nodes,$avg_window_nodes,$avg_dup_nodes_per_window,$avg_dup_ratio_per_window,$TRAIN_PEAK_GPU_MEM_MB,$TRAIN_EPOCH_TIME,$candidate_score,$search_phase,$selection_reason,TRAIN_VALIDATED,$train_log"
    TRAIN_VALIDATE_STATUS="TRAIN_VALIDATED"
    TRAIN_LOG_PATH="$train_log"
    return 0
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

        stage1_file=$(mktemp)
        preproc_candidate_file=$(mktemp)
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
        final_candidate_score=""
        final_search_phase=""
        final_selection_reason=""

        # Stage A1: expansion to find a soft upper band for n_parts
        upper_n=""
        lower_fail_n=""
        current_n="$start_n_parts"
        jump_step="$step_n_parts"
        while [[ "$current_n" -le "$max_n_parts" ]]; do
            stage1_log="$LOG_DIR/${dataset}_${model_alias}_n${current_n}_r2_stage1.log"
            if evaluate_preprocess_candidate "$dataset" "$model_alias" "$attn_type" "$current_n" 2 "$dynamic_target_max_window_nodes" "$start_n_parts" "WINDOW_BOUNDARY" "stage1_scan" "$stage1_log"; then
                printf '%s|%s\n' "$PRE_CANDIDATE_SCORE" "$current_n" >> "$stage1_file"
                if [[ "$PRE_MAX_WINDOW_NODES" -le "$dynamic_target_max_window_nodes" ]]; then
                    upper_n="$current_n"
                    break
                fi
                lower_fail_n="$current_n"
            fi
            if [[ "$current_n" -eq "$max_n_parts" ]]; then
                break
            fi
            next_n=$((current_n + jump_step))
            if [[ "$next_n" -gt "$max_n_parts" ]]; then
                next_n="$max_n_parts"
            fi
            if [[ "$next_n" -eq "$current_n" ]]; then
                break
            fi
            current_n="$next_n"
            jump_step=$((jump_step * 2))
        done

        # Stage A2: binary-ish refinement around the first acceptable upper bound
        if [[ -n "$upper_n" ]]; then
            if [[ -z "$lower_fail_n" ]]; then
                lower_fail_n=$(( upper_n > step_n_parts ? upper_n - step_n_parts : 1 ))
            fi
            while [[ $((upper_n - lower_fail_n)) -gt "$step_n_parts" ]]; do
                mid_n=$(((upper_n + lower_fail_n) / 2))
                if [[ "$mid_n" -le "$lower_fail_n" || "$mid_n" -ge "$upper_n" ]]; then
                    break
                fi
                stage1_log="$LOG_DIR/${dataset}_${model_alias}_n${mid_n}_r2_stage1.log"
                if evaluate_preprocess_candidate "$dataset" "$model_alias" "$attn_type" "$mid_n" 2 "$dynamic_target_max_window_nodes" "$start_n_parts" "WINDOW_BOUNDARY" "stage1_refine" "$stage1_log"; then
                    printf '%s|%s\n' "$PRE_CANDIDATE_SCORE" "$mid_n" >> "$stage1_file"
                    if [[ "$PRE_MAX_WINDOW_NODES" -le "$dynamic_target_max_window_nodes" ]]; then
                        upper_n="$mid_n"
                    else
                        lower_fail_n="$mid_n"
                    fi
                else
                    lower_fail_n="$mid_n"
                fi
            done
        fi

        n_candidates=$(collect_boundary_n_candidates "$stage1_file" "$upper_n" "$step_n_parts" "$max_n_parts")
        if [[ -z "$n_candidates" ]]; then
            final_status="NO_WINDOW_SIZE_MATCH"
        else
            for n_parts in $n_candidates; do
                coarse_file=$(mktemp)
                for related_topk in 1 3 5; do
                    log_path="$LOG_DIR/${dataset}_${model_alias}_n${n_parts}_r${related_topk}.log"
                    if evaluate_preprocess_candidate "$dataset" "$model_alias" "$attn_type" "$n_parts" "$related_topk" "$dynamic_target_max_window_nodes" "$start_n_parts" "R_COARSE" "coarse_scan" "$log_path"; then
                        append_preproc_candidate "$preproc_candidate_file" "$n_parts" "$related_topk" "$PRE_MAX_WINDOW_NODES" "$PRE_AVG_WINDOW_NODES" "$PRE_AVG_DUP_NODES_PER_WINDOW" "$PRE_AVG_DUP_RATIO_PER_WINDOW" "$PRE_CANDIDATE_SCORE" "R_COARSE" "coarse_scan" "$log_path"
                        printf '%s|%s|%s\n' "$PRE_CANDIDATE_SCORE" "$related_topk" "$PRE_AVG_DUP_RATIO_PER_WINDOW" >> "$coarse_file"
                    fi
                done
                adaptive_rs=$(get_adaptive_r_sequence "$coarse_file")
                rm -f "$coarse_file"
                for related_topk in $adaptive_rs; do
                    log_path="$LOG_DIR/${dataset}_${model_alias}_n${n_parts}_r${related_topk}.log"
                    if evaluate_preprocess_candidate "$dataset" "$model_alias" "$attn_type" "$n_parts" "$related_topk" "$dynamic_target_max_window_nodes" "$start_n_parts" "R_REFINE" "adaptive_refine" "$log_path"; then
                        append_preproc_candidate "$preproc_candidate_file" "$n_parts" "$related_topk" "$PRE_MAX_WINDOW_NODES" "$PRE_AVG_WINDOW_NODES" "$PRE_AVG_DUP_NODES_PER_WINDOW" "$PRE_AVG_DUP_RATIO_PER_WINDOW" "$PRE_CANDIDATE_SCORE" "R_REFINE" "adaptive_refine" "$log_path"
                    fi
                done
            done

            top_candidates=$(select_top_candidates "$preproc_candidate_file" "$MAX_TRAIN_VALIDATIONS_PER_CASE")
            had_preproc_candidates=0
            had_dup_in_range=0
            oom_in_train=0
            best_train_time=""
            while IFS='|' read -r candidate_score n_parts related_topk max_window_nodes avg_window_nodes avg_dup_nodes_per_window avg_dup_ratio_per_window search_phase selection_reason log_path; do
                [[ -z "$n_parts" ]] && continue
                had_preproc_candidates=1
                if [[ "$(is_dup_ratio_in_range "$avg_dup_ratio_per_window")" == "1" ]]; then
                    had_dup_in_range=1
                fi
                if validate_train_candidate "$dataset" "$model_alias" "$attn_type" "$n_parts" "$related_topk" "$dynamic_target_max_window_nodes" "$max_window_nodes" "$avg_window_nodes" "$avg_dup_nodes_per_window" "$avg_dup_ratio_per_window" "$candidate_score" "TRAIN_VALIDATE" "selected_for_training"; then
                    take_candidate=0
                    if [[ -z "$best_train_time" ]]; then
                        take_candidate=1
                    elif [[ "$(is_less_than "$TRAIN_EPOCH_TIME" "$best_train_time")" == "1" ]]; then
                        take_candidate=1
                    fi
                    if [[ "$take_candidate" == "1" ]]; then
                        best_train_time="$TRAIN_EPOCH_TIME"
                        final_status="SUCCESS"
                        final_n_parts="$n_parts"
                        final_related_topk="$related_topk"
                        final_max_window="$max_window_nodes"
                        final_avg_window="$avg_window_nodes"
                        final_avg_dup_nodes="$avg_dup_nodes_per_window"
                        final_avg_dup_ratio="$avg_dup_ratio_per_window"
                        final_peak_gpu_mem_mb="$TRAIN_PEAK_GPU_MEM_MB"
                        final_train_epoch_time="$TRAIN_EPOCH_TIME"
                        final_log_path="$TRAIN_LOG_PATH"
                        final_candidate_score="$candidate_score"
                        final_search_phase="TRAIN_VALIDATE"
                        final_selection_reason="best_train_time_under_budget"
                    fi
                else
                    if [[ "$TRAIN_VALIDATE_STATUS" == "OOM_AT_TRAIN" ]]; then
                        oom_in_train=1
                    fi
                fi
            done <<< "$top_candidates"

            if [[ "$final_status" != "SUCCESS" ]]; then
                if [[ "$oom_in_train" == "1" ]]; then
                    final_status="OOM_AT_TRAIN"
                elif [[ "$had_preproc_candidates" == "1" && "$had_dup_in_range" == "1" ]]; then
                    final_status="OOM_AT_TRAIN"
                elif [[ "$had_preproc_candidates" == "1" ]]; then
                    final_status="NO_DUP_RATIO_MATCH"
                    best_preproc=$(sort -t '|' -k1,1g "$preproc_candidate_file" | head -n 1)
                    if [[ -n "$best_preproc" ]]; then
                        IFS='|' read -r final_candidate_score final_n_parts final_related_topk final_max_window final_avg_window final_avg_dup_nodes final_avg_dup_ratio final_search_phase final_selection_reason final_log_path <<< "$best_preproc"
                    fi
                else
                    final_status="NO_WINDOW_SIZE_MATCH"
                    best_stage1=$(sort -t '|' -k1,1g "$stage1_file" | head -n 1)
                    if [[ -n "$best_stage1" ]]; then
                        IFS='|' read -r final_candidate_score final_n_parts <<< "$best_stage1"
                        final_search_phase="WINDOW_BOUNDARY"
                        final_selection_reason="best_stage1_window_candidate"
                    fi
                fi
            fi
        fi

        append_csv_row "$SUMMARY_CSV" "$dataset,$model_alias,$attn_type,$final_n_parts,$final_related_topk,$gpu_total_mem_mb,$gpu_budget_mb,$dynamic_target_max_window_nodes,$final_max_window,$final_avg_window,$final_avg_dup_nodes,$final_avg_dup_ratio,$final_peak_gpu_mem_mb,$final_train_epoch_time,$final_candidate_score,$final_search_phase,$final_selection_reason,$final_status,$final_log_path"
        printf '[AUTOTUNE][SUMMARY] dataset=%s model=%s attn=%s target_max_window=%s budget_mb=%s status=%s n_parts=%s related_topk=%s peak_gpu_mem_mb=%s train_epoch_time=%s candidate_score=%s\n' \
            "$dataset" "$model_alias" "$attn_type" "$dynamic_target_max_window_nodes" "$gpu_budget_mb" "$final_status" "$final_n_parts" "$final_related_topk" "$final_peak_gpu_mem_mb" "$final_train_epoch_time" "$final_candidate_score"

        rm -f "$stage1_file" "$preproc_candidate_file"
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
