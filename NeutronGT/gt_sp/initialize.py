import random
import os
import time
import numpy as np
import torch
from datetime import timedelta


# For DeepSpeed's sequence parallel
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_RANK = None
_SEQUENCE_LENGTH = None

# For sequence parallel rest nodes
_GLOBAL_TOKEN_INDICES = None
_GLOBAL_TOKEN_INDICES_LAST_BATCH = None
_GLOBAL_TOKEN_NUM = None
_REST_SPLIT_SIZES = None
_LAST_BATCH_FLAG = False


def initialize_distributed(args):
    """Initialize torch.distributed and core model parallel."""
    device_count = torch.cuda.device_count()
    assert device_count != 0, 'expected PU number > 0.'
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()                # rank 编号
        args.world_size = torch.distributed.get_world_size()    # rank 数

    else:   
        args.rank = int(os.environ["RANK"])                     
        args.world_size = int(os.environ["WORLD_SIZE"])
        print(f"args.world_size:{args.world_size}")
        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)

        # Manually set the device ids.
        if device_count > 0:
            # device_count: 当前node上 GPU 数量
            # device      : rank 在当前node对应的 GPU 编号 
            device = args.rank % device_count          
            # local_rank  : 所有rank会分到每个node上的每个GPU上,这是本node范围内的 rank 号
            # 由于单卡单进程,local_rk应该等于device号          
            if args.local_rank is not None:                     
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)                       # 这个rank里所有的 tensor.cuda() 操作在这张卡上进行
    
    global _GLOBAL_TOKEN_NUM
    _GLOBAL_TOKEN_NUM = args.num_global_node


    # Call the init process
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        timeout=timedelta(minutes=args.distributed_timeout_minutes))

    # Set the sequence-parallel communicators.
    if device_count > 0:
        args.sequence_parallel_size = args.world_size # sp only 
        initialize_sequence_parallel(args.seq_len, 1, 1,
                                        args.sequence_parallel_size)


def initialize_sequence_parallel(
    seq_length: int,                            # 输入序列的总长度
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    sequence_parallel_size: int = 1,            # 序列并行的规模，即并行处理一个序列需要几个卡
) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()     
    
    if sequence_parallel_size is None:
        sequence_parallel_size = world_size                  
    else:
        assert world_size % sequence_parallel_size == 0
        
    # world_size: 总卡数
    # sequence_parallel_size:一个序列并行需要的卡数
    # num_sequence_parallel_groups: 可以进行几组序列并行
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    
    rank = torch.distributed.get_rank()
    
    # For sequence parallel
    global _SEQUENCE_PARALLEL_GROUP         # 当前进程所属的通信组
    global _SEQUENCE_PARALLEL_WORLD_SIZE    # 通信组内的 rank 数        
    global _SEQUENCE_PARALLEL_RANK          # 当前 rank 在group内的 id
    global _SEQUENCE_LENGTH                 # 总序列长度
    global _SEQUENCE_LENGTH_PER_RANK        # 每张卡分到的子序列长度
    
    # Build the sequence parallel groups.
    _SEQUENCE_LENGTH = seq_length
    assert _SEQUENCE_PARALLEL_GROUP is None, \
    'sequence parallel group is already initialized'
    for i in range(num_sequence_parallel_groups):
        
        # group 内的 rank
        # e.g. group[1]:range(4, 8) -> [4, 5, 6, 7]
        ranks = range(i * sequence_parallel_size,
                        (i + 1) * sequence_parallel_size)
        
        # 为这些rank创建一个新的通信组
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group        
            _SEQUENCE_PARALLEL_RANK = ranks.index(rank)
            _SEQUENCE_PARALLEL_WORLD_SIZE = len(ranks)
            _SEQUENCE_LENGTH_PER_RANK = seq_length // _SEQUENCE_PARALLEL_WORLD_SIZE # for node-level tasks
    
    if torch.distributed.get_rank() == 0:
        print("************ Finish sequence parallel group Initialization. ***********")   
            

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GROUP is not None, \
        'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GROUP

            
def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    if _SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())
    
    
def sequence_parallel_is_initialized():
    """Check if sequence and data parallel groups are initialized."""
    # if _SEQUENCE_PARALLEL_GROUP is None or \
    #     _DATA_PARALLEL_GROUP is None:
    if _SEQUENCE_PARALLEL_GROUP is None:
        return False
    return True


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_parallel_group())


def get_sequence_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the sequence parallel group."""
    global_rank = torch.distributed.get_rank()                  # global GPU 编号
    local_world_size = get_sequence_parallel_world_size()       # 单个group内的 GPU数
    # global_rank // local_world_size: GPU属于的group编号 group_id 
    # group_id * local_world_size = group_src_rank: group内的src_rank号
    return (global_rank // local_world_size) * local_world_size  


def get_sequence_length():
    """Return sequence length for the sequence parallel group."""
    assert _SEQUENCE_LENGTH is not None, \
        'sequence length is not initialized'
    return _SEQUENCE_LENGTH


def get_sequence_length_per_rank():
    """Return local sequence length for the sequence parallel group."""
    assert _SEQUENCE_LENGTH_PER_RANK is not None, \
        'subsequence length is not initialized'
    return _SEQUENCE_LENGTH_PER_RANK


def set_global_token_indices(global_token_indices):
    """Set global token indices for the sequence parallel group."""
    global _GLOBAL_TOKEN_INDICES
    _GLOBAL_TOKEN_INDICES = global_token_indices


def set_last_batch_global_token_indices(global_token_indices_last_batch):
    global _GLOBAL_TOKEN_INDICES_LAST_BATCH
    _GLOBAL_TOKEN_INDICES_LAST_BATCH = global_token_indices_last_batch


def get_global_token_indices(last_batch=False):
    """get global token indices for the sequence parallel group.
    The last batch's nodes are less than seq_len.
    """
    if not last_batch:
        assert _GLOBAL_TOKEN_INDICES is not None, \
        'global token indices is not initialized'
        return _GLOBAL_TOKEN_INDICES
    else:
        return _GLOBAL_TOKEN_INDICES_LAST_BATCH
    

def get_global_token_num():
    """Set global token indices for the sequence parallel group."""
    assert _GLOBAL_TOKEN_NUM is not None, \
        'global token num is not initialized'
    return _GLOBAL_TOKEN_NUM


def last_batch_flag(last_batch=False):
    """Set rest split sizes for the sequence parallel group."""
    global _LAST_BATCH_FLAG
    _LAST_BATCH_FLAG = last_batch
    

def get_last_batch_flag():
    """Set rest split sizes for the sequence parallel group."""
    return _LAST_BATCH_FLAG