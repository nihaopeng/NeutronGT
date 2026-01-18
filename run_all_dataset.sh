#!/bin/bash
GPU_ID=2

# 定义日志保存的文件夹，如果没有会自动创建
LOG_DIR="training_logs"
mkdir -p $LOG_DIR

DATASETS=(
    "citeseer"
    "cora"
    "ogbn-arxiv"
    # "ogbn-products"
    "pubmed"
    "reddit"
    "ogbn-papers100M" 
)


for DATASET in "${DATASETS[@]}"
do
    echo "----------------------------------------------------------------"
    echo "Starting training on dataset: $DATASET"
    echo "Time: $(date)"
    echo "----------------------------------------------------------------"

    # 日志文件名 (例如: log_cora_2024-01-16.txt)
    CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/${DATASET}_${CURRENT_TIME}_origin.log"

    CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=1 --master_port 8081 \
        main_sp_node_level.py \
        --dataset "$DATASET" \
        --seq_len 3200 \
        --n_layers 4 \
        --hidden_dim 64 \
        --ffn_dim 64 \
        --num_heads 8 \
        --epochs 1500 \
        --model gt \
        --distributed-backend 'nccl' \
        --attn_type full \
        --reorder > "$LOG_FILE" 2>&1

    # 检查上一个命令是否执行成功
    if [ $? -eq 0 ]; then
        echo "Dataset $DATASET finished successfully. Log saved to $LOG_FILE"
    else
        echo "Dataset $DATASET failed! Check log at $LOG_FILE"
    fi
    
    echo ""
    # 可选：每个任务间隔几秒，给 GPU 显存一点回收时间
    sleep 5
done

echo "All tasks completed!"