#!/usr/bin/env bash

set -u
set -o pipefail

DEVICES=${1-}
if [ -z "$DEVICES" ] || [[ "$DEVICES" == -* ]]; then
    echo "Error: CUDA_VISIBLE_DEVICES argument is required as the first parameter."
    echo "Usage: bash $0 <devices>"
    echo "Example: bash $0 0,1,2,3"
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_DIR="${ROOT_DIR}/Baseline"
NEUTRONGT_DIR="${ROOT_DIR}/NeutronGT"
DATASET_DIR="${ROOT_DIR}/dataset/"
DATASET="ogbn-papers100M"
RUN_TAG=$(date +"%Y%m%d_%H%M")

export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

device_num=$(echo "$DEVICES" | tr -cd ',' | wc -c)
device_num=$((device_num + 1))

echo "-----------------------------------------------------------------"
echo "System config: Detected $device_num GPUs (Devices: $DEVICES)"
echo "Dataset: ${DATASET}"
echo "Epochs: 10"
echo "-----------------------------------------------------------------"

BASELINE_LOG_DIR="${BASELINE_DIR}/TorchGT_logs"
NEUTRONGT_LOG_DIR="${NEUTRONGT_DIR}/NeutronGT_logs"
mkdir -p "${BASELINE_LOG_DIR}" "${NEUTRONGT_LOG_DIR}"

run_baseline_one() {
    local model_input=$1

    local model_alias=""
    local model_arg=""
    local n_layers=0
    local hidden_dim=0
    local num_heads=0
    local seq_len=0

    case "$model_input" in
        "GT")
            model_alias="GT"
            model_arg="gt"
            n_layers=4
            hidden_dim=128
            num_heads=8
            seq_len=256000
            ;;
        "GPH_Slim")
            model_alias="Graphormer-Slim"
            model_arg="graphormer"
            n_layers=4
            hidden_dim=64
            num_heads=8
            seq_len=256000
            ;;
        "GPH_Large")
            model_alias="Graphormer-Large"
            model_arg="graphormer"
            n_layers=12
            hidden_dim=768
            num_heads=32
            seq_len=32000
            ;;
        *)
            echo "Error: unsupported baseline model: $model_input"
            return 1
            ;;
    esac

    local ffn_dim=$hidden_dim
    local log_file="${BASELINE_LOG_DIR}/${DATASET}_${model_alias}_Sparse_Reorder_10ep_${RUN_TAG}.log"
    local master_port=$((8000 + RANDOM % 1000))

    echo "-----------------------------------------------------------------"
    echo "Framework: Baseline"
    echo "Model: ${model_alias}"
    echo "Params: Epochs=10, Seq_Len=${seq_len}, Port=${master_port}"
    echo "Log file: ${log_file}"
    echo "-----------------------------------------------------------------"

    (
        cd "${BASELINE_DIR}" && \
        CUDA_VISIBLE_DEVICES="${DEVICES}" torchrun \
            --nproc_per_node="${device_num}" \
            --master_port="${master_port}" \
            main_sp_node_level.py \
            --dataset_dir "${DATASET_DIR}" \
            --dataset "${DATASET}" \
            --seq_len "${seq_len}" \
            --n_layers "${n_layers}" \
            --hidden_dim "${hidden_dim}" \
            --ffn_dim "${ffn_dim}" \
            --num_heads "${num_heads}" \
            --epochs 10 \
            --model "${model_arg}" \
            --distributed-backend nccl \
            --attn_type sparse \
            --distributed-timeout-minutes 120 \
            --reorder
    ) > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        echo "Status: Success"
    else
        echo "Status: Failed. Check ${log_file}"
        return 1
    fi
}

run_neutrongt_one() {
    local model_input=$1

    local model_alias=""
    local model_arg=""
    local n_layers=0
    local hidden_dim=0
    local num_heads=0
    local n_parts=0
    local related_topk=0

    case "$model_input" in
        "GT")
            model_alias="GT"
            model_arg="gt_sw"
            n_layers=4
            hidden_dim=128
            num_heads=8
            n_parts=4096
            related_topk=2
            ;;
        "GPH_Slim")
            model_alias="GPH_Slim"
            model_arg="graphormer"
            n_layers=4
            hidden_dim=64
            num_heads=8
            n_parts=4096
            related_topk=2
            ;;
        "GPH_Large")
            model_alias="GPH_Large"
            model_arg="graphormer"
            n_layers=12
            hidden_dim=768
            num_heads=32
            n_parts=8192
            related_topk=2
            ;;
        *)
            echo "Error: unsupported NeutronGT model: $model_input"
            return 1
            ;;
    esac

    local ffn_dim=$hidden_dim
    local log_file="${NEUTRONGT_LOG_DIR}/${DATASET}_${model_alias}_nparts${n_parts}_rtopk${related_topk}_pprtopk5_10ep_${RUN_TAG}.log"
    local master_port=$((9000 + RANDOM % 1000))

    echo "-----------------------------------------------------------------"
    echo "Framework: NeutronGT"
    echo "Model: ${model_alias}"
    echo "Params: Epochs=10, NPARTS=${n_parts}, RELATED_TOPK=${related_topk}, PPR_TOPK=5, PPR_ITER_TOPK=5, Port=${master_port}"
    echo "Log file: ${log_file}"
    echo "-----------------------------------------------------------------"

    (
        cd "${NEUTRONGT_DIR}" && \
        CUDA_VISIBLE_DEVICES="${DEVICES}" torchrun \
            --nproc_per_node="${device_num}" \
            --master_port="${master_port}" \
            main_sp_node_level_ppr.py \
            --dataset "${DATASET}" \
            --dataset_dir "${DATASET_DIR}" \
            --model "${model_arg}" \
            --attn_type sparse \
            --n_layers "${n_layers}" \
            --hidden_dim "${hidden_dim}" \
            --ffn_dim "${ffn_dim}" \
            --num_heads "${num_heads}" \
            --epochs 10 \
            --use_cache 1 \
            --use_preprocess_cache 0 \
            --struct_enc False \
            --n_parts "${n_parts}" \
            --related_nodes_topk_rate "${related_topk}" \
            --ppr_backend appnp \
            --ppr_topk 5 \
            --ppr_alpha 0.85 \
            --ppr_num_iterations 10 \
            --ppr_batch_size 8192 \
            --ppr_iter_topk 5 \
            --distributed-backend nccl \
            --distributed-timeout-minutes 120
    ) > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        echo "Status: Success"
    else
        echo "Status: Failed. Check ${log_file}"
        return 1
    fi
}

run_baseline_one "GT"
run_baseline_one "GPH_Slim"
run_baseline_one "GPH_Large"

run_neutrongt_one "GT"
run_neutrongt_one "GPH_Slim"
run_neutrongt_one "GPH_Large"
