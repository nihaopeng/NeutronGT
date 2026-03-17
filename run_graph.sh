#!/bin/bash

# main_sp_sparse_malnet.py 使用 mp.Process 手动启动子进程，但脚本被 torchrun 启动，导致冲突。所以
# - main_sp_sparse_malnet.py 应该 不使用 torchrun 直接运行
# - main_sp_graph_level.py 使用 torchrun 运行


IFS="," read -ra arr <<< "$1"
device_num=${#arr[@]}
echo "显卡数量: ${device_num}，准备 Graph-level MalNet 分布式训练..."

CUDA_VISIBLE_DEVICES=$1 python  main_sp_sparse_malnet.py \
    --dataset MalNetTiny \
    --model graphormer \
    --seq_len 512 \
    --batch_size 16 \
    --n_layers 4 \
    --hidden_dim 64 \
    --num_heads 8 \
    --ffn_dim 64 \
    --epochs 100 \
    --attn_type sparse \
    --peak_lr 2e-4 \
    --sequence-parallel-size ${device_num}

# CUDA_VISIBLE_DEVICES=$1 torchrun \
#     --nproc_per_node=${device_num} \
#     --master_port 8082 \
#     main_sp_graph_level.py \
#     --dataset MalNet \
#     --batch_size 128 \
#     --seq_len 128 \
#     --n_layers 4 \
#     --hidden_dim 64 \
#     --ffn_dim 64 \
#     --num_heads 8 \
#     --epochs 2000 \
#     --model graphormer \
#     --distributed-backend 'nccl' \
#     --attn_type sparse \
#     --num_workers 4 \
#     --edge_type multi_hop \
#     --multi_hop_max_dist 20 \
#     --spatial_pos_max 1024 --reorder

