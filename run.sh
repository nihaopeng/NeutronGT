
IFS="," read -ra arr <<< "$1"
device_num=${#arr[@]}
echo "显卡数量: ${device_num}，准备启动分布式训练..."

export LD_LIBRARY_PATH=/home/pengyt/softwares/miniconda/envs/gt/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=${device_num} --master_port 8090 main_sp_node_level.py --dataset ogbn-arxiv --seq_len 64000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 2000 --model gt --distributed-backend 'nccl' --attn_type flash 

# CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=${device_num} --master_port 8090 main_sp_node_level_full_batch.py --dataset ogbn-arxiv --seq_len 64000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 2000 --model gt --distributed-backend nccl --attn_type flash   --full_batch
