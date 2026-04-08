export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 29652 main_sp_node_level_ppr.py \
  --dataset $1 \
  --dataset_dir ./dataset/ \
  --model graphormer \
  --attn_type $2 \
  --n_layers 4 \
  --hidden_dim 64 \
  --ffn_dim 64 \
  --num_heads 8 \
  --epochs 50 \
  --seq_len 64000 \
  --struct_enc False \
  --max_dist 5 \
  --use_cache $3 \
  --num_global_node 1 \
  --n_parts 128 \
  --related_nodes_topk_rate 12 \
  --ppr_backend appnp \
  --ppr_topk 5 \
  --ppr_alpha 0.85 \
  --ppr_num_iterations 10 \
  --ppr_batch_size 8192 \
  --ppr_iter_topk 5 \
  --distributed-backend nccl \
  --distributed-timeout-minutes 120
