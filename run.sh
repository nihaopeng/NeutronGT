## multi npu
IFS="," read -ra arr <<< "$2"
device_num=${#arr[@]}
# echo ${device_num}
case $1 in
"metis")
echo "The metis entry is deprecated. Use the ppr entry instead."
exit 1
;;
"ppr")
# an example: bash run.sh ppr 0,1,2,3 8012 cora gt_sw True 5
# CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=${device_num} --master_port $3 main_sp_node_level_ppr.py --dataset $4 --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1000 --model $5 --distributed-backend 'nccl' --attn_type 'sparse' --reorder --struct_enc $6 --max_dist $7   --use_cache 1
# CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=${device_num} --master_port $3 main_sp_node_level_ppr.py --dataset $4 --seq_len 6400 --n_layers 12 --hidden_dim 768 --ffn_dim 768 --num_heads 32 --epochs 1000 --model $5 --distributed-backend 'nccl' --attn_type 'full' --reorder --struct_enc $6 --max_dist $7  --vis_dir $8 --use_cache 1
CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=${device_num} --master_port $3 main_sp_node_level_ppr.py --dataset $4 --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1000 --model $5 --distributed-backend 'nccl' --attn_type 'sparse' --reorder --struct_enc $6 --max_dist $7   --use_cache 1 --ppr_backend appnp --ppr_topk 5 --ppr_alpha 0.85 --ppr_num_iterations 10 --ppr_batch_size 512 --ppr_iter_topk 50
;;
"origin")
CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=${device_num} --master_port $3 main_sp_node_level.py --dataset $4 --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1000 --model $5 --distributed-backend 'nccl' --attn_type 'full' --reorder --struct_enc $6 --max_dist $7   --use_cache 0
;;
esac
