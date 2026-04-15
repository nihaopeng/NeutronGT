#!/bin/bash

DEVICES=$1
if [ -z "$DEVICES" ]; then
    echo "Error: CUDA_VISIBLE_DEVICES argument is required."
    echo "Usage: bash $0 <devices> (e.g., bash $0 0,1,2,3)"
    exit 1
fi

# Calculate the number of GPUs based on the input string
device_num=$(echo $DEVICES | tr -cd ',' | wc -c)
device_num=$((device_num + 1))

echo "-----------------------------------------------------------------"
echo "System config: Detected $device_num GPUs (Devices: $DEVICES)"
echo "-----------------------------------------------------------------"

CURRENT_DATE=$(date +"%Y%m%d_%H%M")
LOG_DIR="experiment_logs_origin"
mkdir -p ${LOG_DIR}

# ================= Configuration =================
# Dataset list
DATASETS=(
    "ogbn-products"
    # "AmazonProducts"
    # "ogbn-arxiv"
    # "ogbn-papers100M"
    # "reddit"
)

# Model configs: "Alias|model_arg|n_layers|hidden_dim|num_heads"
MODELS=(
    # "Graphormer-Slim|graphormer|4|64|8"
    "GT|gt|4|128|8"
    # "Graphormer-Large|graphormer|12|768|32"
)

# Attention strategies: "Alias|attn_type|use_reorder(true/false)"
ATTENTIONS=(
    "Sparse_Reorder|sparse|true"
)

# ================= Main Execution =================
for dataset in "${DATASETS[@]}"; do
    for model_info in "${MODELS[@]}"; do
        
        # Parse model parameters
        IFS='|' read -r model_alias model_arg n_layers hidden_dim num_heads <<< "$model_info"
        
        # Sanity check for empty parsing (e.g., due to hidden characters or CRLF)
        if [ -z "$model_alias" ] || [ -z "$num_heads" ]; then
            echo "Error: Failed to parse model configuration: $model_info"
            exit 1
        fi

        ffn_dim=$hidden_dim

        # Determine sequence length and epochs based on model type
        SEQ_LEN=0
        CURRENT_EPOCHS=200 # Default fallback
        
        if [ "$model_alias" == "Graphormer-Slim" ] || [ "$model_alias" == "GT" ]; then
            SEQ_LEN=256000
            CURRENT_EPOCHS=500
        elif [ "$model_alias" == "Graphormer-Large" ]; then
            SEQ_LEN=32000
            CURRENT_EPOCHS=200
        fi

        for attn_info in "${ATTENTIONS[@]}"; do
            
            IFS='|' read -r strategy_alias attn_type use_reorder <<< "$attn_info"

            LOG_FILE="${LOG_DIR}/${dataset}_${model_alias}_${strategy_alias}_${CURRENT_DATE}.log"

            # Generate random port to avoid TIME_WAIT conflicts during restarts
            MASTER_PORT=$(( 8000 + RANDOM % 1000 ))

            echo "Starting experiment | Dataset: $dataset | Model: $model_alias | Strategy: $strategy_alias"
            echo "Params: Epochs=${CURRENT_EPOCHS}, Seq_Len=${SEQ_LEN}, Port=${MASTER_PORT}"
            echo "Log file: $LOG_FILE"

            # Construct torchrun arguments via array for robust parsing
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
                --epochs "${CURRENT_EPOCHS}"
                --model "${model_arg}"
                --distributed-backend "nccl"
                --attn_type "${attn_type}"
                --distributed-timeout-minutes 120
            )

            # Dynamically append boolean flags
            if [ "$use_reorder" = "true" ]; then
                TORCHRUN_ARGS+=(--reorder)
            fi

            # Execute distributed training
            CUDA_VISIBLE_DEVICES=$DEVICES torchrun "${TORCHRUN_ARGS[@]}" > "${LOG_FILE}" 2>&1

            if [ $? -eq 0 ]; then
                echo "Status: Success"
            else
                echo "Status: Failed (Exit code: $?). Please check $LOG_FILE for details."
            fi
            echo "-----------------------------------------------------------------"
            
            sleep 3

        done
    done
done

echo "All scheduled experiments have been completed."