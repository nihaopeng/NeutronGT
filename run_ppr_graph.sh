#!/bin/bash

# ==============================================================================
# TorchGT MalNet Tiny - PPR + METIS 分区训练脚本
# ==============================================================================
# 
# 功能：
#   - 使用 PPR + METIS 对图进行分区
#   - 支持 KV Cache 优化
#   - 支持 GT 和 Graphormer 模型
#
# 使用方法：
#   bash scripts/train_malnet_ppr.sh <gpu_ids>
#   
#   示例：
#     bash scripts/train_malnet_ppr.sh 0,1,2,3
#     bash scripts/train_malnet_ppr.sh 0,1
#     bash scripts/train_malnet_ppr.sh 0
#
# ==============================================================================

# 检查参数
if [ -z "$1" ]; then
    echo "Usage: bash scripts/train_malnet_ppr.sh <gpu_ids>"
    echo "Example: bash scripts/train_malnet_ppr.sh 0,1,2,3"
    exit 1
fi

# 解析 GPU 数量
IFS="," read -ra arr <<< "$1"
device_num=${#arr[@]}
echo "=========================================="
echo "显卡数量: ${device_num}"
echo "GPU IDs: $1"
echo "TorchGT MalNet PPR Training"
echo "=========================================="

# 模型类型 (gt 或 graphormer)，默认使用 graphormer
MODEL=${2:-"graphormer"}

# 数据集
DATASET="MalNetTiny"

# 模型参数
N_LAYERS=4
HIDDEN_DIM=64
FFN_DIM=64
NUM_HEADS=8

# 训练参数
SEQ_LEN=512
EPOCHS=100
BATCH_SIZE=16
LR=2e-4

# 注意力类型 (sparse, full, flash)
ATTN_TYPE="sparse"

# 端口
PORT=8081



echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Attn Type: ${ATTN_TYPE}"
echo "=========================================="

# 运行训练
CUDA_VISIBLE_DEVICES=$1  torchrun --nproc_per_node=${device_num}  main_sp_sparse_malnet_ppr.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --seq_len ${SEQ_LEN} \
    --n_layers ${N_LAYERS} \
    --hidden_dim ${HIDDEN_DIM} \
    --ffn_dim ${FFN_DIM} \
    --num_heads ${NUM_HEADS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --peak_lr ${LR} \
    --attn_type ${ATTN_TYPE} \

