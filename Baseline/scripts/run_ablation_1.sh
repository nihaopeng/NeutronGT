#!/bin/bash

# 1. Parse the positional argument for GPUs
DEVICES=$1
if [ -z "$DEVICES" ] || [[ "$DEVICES" == -* ]]; then
    echo "Error: CUDA_VISIBLE_DEVICES argument is required as the first parameter."
    echo "Usage: bash $0 <devices> [dataset_flag] [model_flag]"
    echo "Datasets : --products | --amazon | --arxiv | --reddit"
    echo "Models   : --GT | --GPH_Slim | --GPH_Large"
    echo "Example  : bash $0 0,1,2,3 --arxiv --GT"
    exit 1
fi
shift

DATASET_INPUT=""
MODEL_INPUT=""

# 2. Parse long flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --products)  DATASET_INPUT="ogbn-products" ;;
        --amazon)    DATASET_INPUT="AmazonProducts" ;;
        --arxiv)     DATASET_INPUT="ogbn-arxiv" ;;
        --reddit)    DATASET_INPUT="reddit" ;;
        --GT)        MODEL_INPUT="GT" ;;
        --GPH_Slim)  MODEL_INPUT="GPH_Slim" ;;
        --GPH_Large) MODEL_INPUT="GPH_Large" ;;
        *)
            echo "Error: Unknown parameter '$1'"
            echo "Supported datasets: --products, --amazon, --arxiv, --reddit"
            echo "Supported models: --GT, --GPH_Slim, --GPH_Large"
            exit 1
            ;;
    esac
    shift
done

# 3. Validate that both a dataset and a model were selected
if [ -z "$DATASET_INPUT" ] || [ -z "$MODEL_INPUT" ]; then
    echo "Error: You must specify both a dataset and a model."
    echo "Example: bash $0 0,1,2,3 --arxiv --GT"
    exit 1
fi

# ================= Parameter Mapping =================

dataset="$DATASET_INPUT"

case "$MODEL_INPUT" in
    "GT")
        model_info="GT|gt|4|128|8|500"
        ;;
    "GPH_Slim")
        model_info="Graphormer-Slim|graphormer|4|64|8|500"
        ;;
    "GPH_Large")
        model_info="Graphormer-Large|graphormer|12|768|32|200"
        ;;
esac

# Hardcoded ablation strategy: full attention without reorder
attn_info="Full_NoReorder|full|false"

# ================= Main Execution =================

device_num=$(echo $DEVICES | tr -cd ',' | wc -c)
device_num=$((device_num + 1))

echo "-----------------------------------------------------------------"
echo "System config: Detected $device_num GPUs (Devices: $DEVICES)"
echo "-----------------------------------------------------------------"

CURRENT_DATE=$(date +"%Y%m%d_%H%M")
LOG_DIR="TorchGT_logs"
mkdir -p ${LOG_DIR}

IFS='|' read -r model_alias model_arg n_layers hidden_dim num_heads current_epochs <<< "$model_info"
IFS='|' read -r strategy_alias attn_type use_reorder <<< "$attn_info"

ffn_dim=$hidden_dim
SEQ_LEN=16000

LOG_FILE="${LOG_DIR}/${dataset}_${model_alias}_${strategy_alias}_${CURRENT_DATE}.log"
MASTER_PORT=$(( 8000 + RANDOM % 1000 ))

echo "Starting experiment | Dataset: $dataset | Model: $model_alias | Strategy: $strategy_alias"
echo "Params: Epochs=${current_epochs}, Seq_Len=${SEQ_LEN}, Port=${MASTER_PORT}"
echo "Log file: $LOG_FILE"

TORCHRUN_ARGS=(
    --nproc_per_node="${device_num}"
    --master_port="${MASTER_PORT}"
    main_sp_node_level.py
    --dataset "${dataset}"
    --seq_len "${SEQ_LEN}"
    --n_layers "${n_layers}"
    --hidden_dim "${hidden_dim}"
    --ffn_dim "${ffn_dim}"
    --num_heads "${num_heads}"
    --epochs "${current_epochs}"
    --model "${model_arg}"
    --distributed-backend "nccl"
    --attn_type "${attn_type}"
    --distributed-timeout-minutes 120
)

if [ "$use_reorder" = "true" ]; then
    TORCHRUN_ARGS+=(--reorder)
fi

CUDA_VISIBLE_DEVICES=$DEVICES torchrun "${TORCHRUN_ARGS[@]}" > "${LOG_FILE}" 2>&1

if [ $? -eq 0 ]; then
    echo "Status: Success"
else
    echo "Status: Failed (Exit code: $?). Please check $LOG_FILE for details."
fi
echo "-----------------------------------------------------------------"
