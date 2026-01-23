
# full attn
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port 8012 main_sp_node_level_ppr.py --dataset "cora" --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1500 --model 'gt_sw' --distributed-backend 'nccl' --reorder --struct_enc 'True' --max_dist 5 --vis_dir "./vis" --attn_type "full"

# sparse attn
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port 8012 main_sp_node_level_ppr.py --dataset "cora" --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1500 --model 'gt_sw' --distributed-backend 'nccl' --reorder --struct_enc 'True' --max_dist 5 --vis_dir "./vis" --attn_type "sparse"