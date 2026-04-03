import argparse


def parser_add_main_args(parser):
   
   
    parser.add_argument('--enable_attention_pruning', action='store_true', 
                    help='是否开启注意力剪枝实验 (Attention Pruning)')
    parser.add_argument('--attention_pruning_ratio', type=float, default=0.1, 
                        help='剪枝保留比例 (例如 0.1 代表只保留 Top 10% 的高分连接)')
    
    parser.add_argument('--enable_neighbor_pruning', action='store_true', 
                    help='是否开启邻居剪枝实验 (Attention Pruning)')
    parser.add_argument('--neighbor_pruning_ratio', type=float, default=0.1, 
                        help='高分邻居保留比例 (例如 0.1 代表只保留 Top 10% 的高分邻居)')

    
    # main args
    parser.add_argument('--device', type=int, default=0, help='device id')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--dataset', type=str, default='pubmed')

    # model args
    parser.add_argument('--model', type=str, default="graphormer")
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--attn_bias_dim', type=int, default=1) # must match M power adj in preprocess_data
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--input_dropout_rate', type=float, default=0.1)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_global_node', type=int, default=1)
    parser.add_argument('--attn_type', type=str, default="full", help='whether to use sparse attention')

    # training args
    parser.add_argument('--seq_len', type=int, default=256000, help='total sequence length here')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_updates', type=int, default=10,
                        help='warmup steps for optimizer learning rate scheduling')
    parser.add_argument('--tot_updates',  type=int, default=70,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--epochs', type=int, default=100) # larger seq len more training epochs
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    parser.add_argument('--peak_lr', type=float, default=1e-4)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--perturb_feature', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False, help='whether to save model')
    parser.add_argument('--load_model', action='store_true', default=False, help='whether to load saved model')
    parser.add_argument('--model_dir', type=str, default='./model_ckpt/')
    parser.add_argument('--switch_freq', type=int, default=5)
    parser.add_argument('--reorder', action='store_true', default=False,
                        help='TorchGT mode')
    parser.add_argument('--struct_enc', type=str, default="False", help='whether to enable structure encoding')
    parser.add_argument('--max_dist', type=int, default=5)
    parser.add_argument('--max_num_edges', type=int, default=512)
    parser.add_argument('--vis_dir', type=str, help='path of vis')
    parser.add_argument('--use_cache',default=0, type=int,choices=[0,1], help='path of vis')
    parser.add_argument('--ppr_backend', type=str, default='torch_geometric', choices=['torch_geometric', 'appnp'],
                       help='backend used to materialize PPR edges')
    parser.add_argument('--ppr_alpha', type=float, default=0.85,
                       help='teleport coefficient used by the selected PPR backend')
    parser.add_argument('--ppr_topk', type=int, default=5,
                       help='top-k PPR neighbors kept for each source node')
    parser.add_argument('--ppr_num_iterations', type=int, default=10,
                       help='APPNP backend only: number of power-iteration propagation steps')
    parser.add_argument('--ppr_batch_size', type=int, default=8,
                       help='APPNP backend only: number of seed nodes processed per SpMM batch')
    parser.add_argument('--ppr_iter_topk', type=int, default=0,
                       help='APPNP backend only: iterative top-k pruning per propagation step; positive values are strongly recommended for large graphs, while 0 or negative disables pruning and may cause high memory usage')
    parser.add_argument('--ppr_eps', type=float, default=1e-6,
                       help='torch_geometric backend only: epsilon used by get_ppr')
    parser.add_argument('--n_parts', type=int, default=50,
                       help='number of graph partitions/windows used during Metis partitioning')
    
    # distributed args
    parser.add_argument('--rank', type=int, default=None,
                       help='rank passed from distributed launcher.')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    parser.add_argument('--world-size', type=int, default=None,
                       help='world size of sequence parallel group.')
    parser.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo', 'ccl', 'hccl'],
                       help='Which backend to use for distributed training.')
    parser.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    parser.add_argument('--sequence-parallel-size', type=int, default=4,
                       help='Enable DeepSpeed\'s sequence parallel.')