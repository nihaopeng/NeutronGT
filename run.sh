## multi npu
IFS="," read -ra arr <<< "$2"
device_num=${#arr[@]}
# echo ${device_num}
case $1 in
"metis")
CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=${device_num} --master_port $7 main_sp_node_level_metis.py --dataset $3 --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1500 --model gt --distributed-backend 'nccl' --reorder --struct_enc $4 --max_dist $5 --vis_dir $6
;;
"ppr")
CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=${device_num} --master_port $7 main_sp_node_level_ppr.py --dataset $3 --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1500 --model gt --distributed-backend 'nccl' --attn_type 'full' --reorder --struct_enc $4 --max_dist $5  --vis_dir $6
;;
"origin")
CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=${device_num} --master_port $7 main_sp_node_level.py --dataset $3 --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1500 --model gt --distributed-backend 'nccl' --attn_type 'full' --reorder --struct_enc $4 --max_dist $5  --vis_dir $6
;;
esac