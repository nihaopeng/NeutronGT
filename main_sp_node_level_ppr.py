import argparse
import os
import random
import time

import numpy as np
import torch

from gt_sp.initialize import (
    get_sequence_length_per_rank,
    get_sequence_parallel_world_size,
    initialize_distributed,
    sequence_parallel_is_initialized,
    set_global_token_indices,
    set_last_batch_global_token_indices,
)
from gt_sp.reducer import sync_params_and_buffers
from gt_sp.utils import LossStagnationDetector, random_split_idx
from utils.lr import PolynomialDecayLR
from utils.parser_node_level import parser_add_main_args
import utils.logger as logger
import utils.vis as vis
from core.node_level_pipeline import (
    StructInfo,
    broadcast_window_state,
    build_graph_struct_info,
    build_local_partitions,
    build_model,
    eval_epoch,
    load_optional_edge_csr,
    restore_global_window_state,
    sync_device,
    train_epoch,
)


def main():
    logger.IS_LOGGING = False
    parser = argparse.ArgumentParser(description='TorchGT node-level training arguments.')
    parser_add_main_args(parser)
    args = parser.parse_args()

    vis.vis_dir = args.vis_dir
    
    initialize_distributed(args)
    device = f'cuda:{torch.cuda.current_device()}' 
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)
    
    feature = torch.load(args.dataset_dir + args.dataset + '/x.pt')
    y = torch.load(args.dataset_dir + args.dataset + '/y.pt')
    edge_csr_data = load_optional_edge_csr(args.dataset_dir, args.dataset) if args.ppr_backend == 'appnp' else None
    edge_index = None if edge_csr_data is not None else torch.load(args.dataset_dir + args.dataset + '/edge_index.pt')
    N = feature.shape[0]

    if args.dataset == 'pokec':
        y = torch.clamp(y, min=0) 
    if args.dataset == "ogbn-papers100M":
        split_idx_path = os.path.join(args.dataset_dir, args.dataset, "split_idx.pt")
        local_split_idx = torch.load(split_idx_path, map_location="cpu")
        if "train" in local_split_idx and "valid" in local_split_idx and "test" in local_split_idx:
            split_idx = {
                "train": torch.as_tensor(local_split_idx["train"], dtype=torch.long),
                "valid": torch.as_tensor(local_split_idx["valid"], dtype=torch.long),
                "test": torch.as_tensor(local_split_idx["test"], dtype=torch.long),
            }
        elif "train_idx" in local_split_idx and "val_idx" in local_split_idx and "test_idx" in local_split_idx:
            split_idx = {
                "train": torch.as_tensor(local_split_idx["train_idx"], dtype=torch.long),
                "valid": torch.as_tensor(local_split_idx["val_idx"], dtype=torch.long),
                "test": torch.as_tensor(local_split_idx["test_idx"], dtype=torch.long),
            }
        else:
            raise KeyError(f"Unsupported split_idx.pt format in {split_idx_path}: {list(local_split_idx.keys())}")
    else:
        split_idx = random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)

    if args.rank == 0:
        print(args)
        print('Dataset load successfully')
        if edge_csr_data is not None:
            print('APPNP backend will use edge_index_csr.pt')
        print(f"Train nodes: {split_idx['train'].shape[0]}, Val nodes: {split_idx['valid'].shape[0]}, Test nodes: {split_idx['test'].shape[0]}") 
        print(f"Training iters: {split_idx['train'].size(0) // args.seq_len + 1}, Val iters: {split_idx['valid'].size(0) // args.seq_len + 1}, Test iters: {split_idx['test'].size(0) // args.seq_len + 1}")
    
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1

    train_idx = split_idx['train']
    
    if args.rank == 0:
        flatten_train_idx = train_idx.to('cuda')
    else:
        total_numel = train_idx.numel()
        flatten_train_idx = torch.empty(total_numel,
                                device=device,
                                dtype=torch.int64)

    seq_len_per_rank = get_sequence_length_per_rank()
    sub_real_seq_len = seq_len_per_rank + args.num_global_node
    global_token_indices = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))

    if flatten_train_idx.shape[0] % args.seq_len != 0:
        last_batch_node_num = flatten_train_idx.shape[0] % args.seq_len
        if last_batch_node_num % seq_parallel_world_size != 0:
            div = last_batch_node_num // seq_parallel_world_size
            last_batch_node_num = div * seq_parallel_world_size + (seq_parallel_world_size - 1)

        x_dummy_list = [t for t in torch.tensor_split(
            torch.zeros(last_batch_node_num, ), seq_parallel_world_size, dim=0)]
        sub_split_seq_lens = [t.shape[0] for t in x_dummy_list]
        sub_real_seq_len = max(sub_split_seq_lens) + args.num_global_node
        global_token_indices_last_batch = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))
    else:
        sub_split_seq_lens = None
        global_token_indices_last_batch = None
    set_global_token_indices(global_token_indices)
    set_last_batch_global_token_indices(global_token_indices_last_batch)

    sync_device(device)
    graph_preprocess_start = time.time()
    structInfo:StructInfo = build_graph_struct_info(
            args,N,edge_index,feature,seq_parallel_world_size,device,
            topk=args.ppr_topk,
            n_parts=args.n_parts,
            related_nodes_topk_rate=args.related_nodes_topk_rate,
            edge_csr_data=edge_csr_data
        )
    sync_device(device)
    graph_preprocess_total_time = time.time() - graph_preprocess_start
    wm:weightMetis_keepParent = structInfo.wm

    window_state_timing = broadcast_window_state(args, structInfo, feature, device)
    sync_device(device)
    if args.rank == 0:
        print("====================================================================================")
        print("Graph preprocess total time: {:.3f}s".format(graph_preprocess_total_time))
        print("Window state distribution total time: {:.3f}s".format(window_state_timing['window_state_total_time']))
        print("Window bundle write time: {:.3f}s".format(window_state_timing['bundle_write_time']))
        print("====================================================================================")
    print(
        f"[rank {args.rank}] Window rebuild: load={window_state_timing['bundle_load_time']:.3f}s, "
        f"dup={window_state_timing['local_dup_cache_rebuild_time']:.3f}s, "
        f"subgraph={window_state_timing['local_subgraph_rebuild_time']:.3f}s, "
        f"spatial={window_state_timing['local_spatial_rebuild_time']:.3f}s, "
        f"total={window_state_timing['window_state_total_time']:.3f}s"
    )

    local_partition_ids, local_partitions = build_local_partitions(structInfo, args.rank, seq_parallel_world_size)
    if args.use_cache and args.rank == 0:
        total_local_dup_nodes = int(sum(int(part.numel()) for part in structInfo.local_dup_nodes_per_partition))
        unique_local_dup_nodes = int(structInfo.local_dup_nodes_per_partition_feature.shape[0]) if structInfo.local_dup_nodes_per_partition_feature is not None else 0
        local_window_count = len(local_partition_ids)
        avg_dup_nodes_per_window = (total_local_dup_nodes / local_window_count) if local_window_count > 0 else 0.0
        avg_unique_dup_nodes_per_window = (unique_local_dup_nodes / local_window_count) if local_window_count > 0 else 0.0
        print("\n" + "="*80)
        print("本地重复节点缓存统计:")
        print("="*80)
        print(f"rank {args.rank} local partitions: {local_window_count}")
        print(f"  - 本地重复节点出现次数: {total_local_dup_nodes}")
        print(f"  - 本地唯一重复节点数: {unique_local_dup_nodes}")
        print(f"  - 每个窗口平均重复节点个数: {avg_dup_nodes_per_window:.2f}")
        print(f"  - 每个窗口平均唯一重复节点个数: {avg_unique_dup_nodes_per_window:.2f}")
    print(f"rank {args.rank} local_partition_ids: {local_partition_ids}")
    if args.preprocess_only == 1:
        if seq_parallel_world_size > 1:
            torch.distributed.barrier()
        if args.rank == 0:
            print("Preprocess-only mode enabled, exiting before model build/training.")
        return

    model = build_model(args,feature,device,y,graph_in_degree=structInfo.graph_in_degree,graph_out_degree=structInfo.graph_out_degree)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_updates,
            tot=args.epochs,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0
    )
    
    if args.rank == 0:
        print('Model params:', sum(p.numel() for p in model.parameters()))

    if seq_parallel_world_size > 1:
        sync_params_and_buffers(model)

    loss_mean_list = []
    detector = LossStagnationDetector(cooldown=0)
    
    for epoch in range(0, args.epochs):
        loss_mean,scores_by_pid,updated_kv_cache,time_stats = train_epoch(args,model,local_partition_ids,local_partitions,feature,y,optimizer,lr_scheduler,seq_parallel_world_size,split_idx,device,
                    epoch=epoch,
                    structInfo=structInfo)
        
        if epoch % 20 == 0:
            if args.rank == 0:
                print(f"epoch {epoch}: lr = {lr_scheduler.get_last_lr()[0]:.2e}")
            eval_epoch(args,model,local_partition_ids,local_partitions,feature,y,split_idx,device,
                    epoch=epoch,
                    structInfo=structInfo)
        
        loss_mean_list.append(loss_mean)
        if args.use_cache==0 and detector(loss_mean_list):
            if seq_parallel_world_size > 1:
                dist.barrier()
            sync_device(device)
            window_adjust_start = time.time()
            print("!node in and out!")
            gathered_scores = [None for _ in range(args.world_size)]
            dist.all_gather_object(gathered_scores, scores_by_pid)
            if args.rank == 0:
                restore_global_window_state(structInfo)
                merged_scores_by_pid = {}
                for item in gathered_scores:
                    merged_scores_by_pid.update(item)
                ordered_scores = [merged_scores_by_pid[pid] for pid in range(len(wm.partitioned_results))]
                wm.node_out(ordered_scores,remove_ratio=0.3)
                wm.node_in()
            if args.rank == 0:
                structInfo.window_state_version += 1
            broadcast_window_state(args, structInfo, feature, device)
            local_partition_ids, local_partitions = build_local_partitions(structInfo, args.rank, seq_parallel_world_size)
            if args.rank == 0 and hasattr(wm, 'dup_nodes_per_partition'):
                print(f"\n[窗口调整后] 重复节点统计:")
                total_dup_nodes = sum(len(dup_nodes) for dup_nodes in wm.dup_nodes_per_partition)
                non_empty_dup_nodes = [dup for dup in wm.dup_nodes_per_partition if len(dup) > 0]
                unique_dup_nodes = len(torch.unique(torch.cat(non_empty_dup_nodes))) if non_empty_dup_nodes else 0
                total_partition_nodes = sum(len(partition) for partition in wm.partitioned_results)
                print(f"  - 总分区节点数: {total_partition_nodes}")
                print(f"  - 重复节点出现次数: {total_dup_nodes}")
                print(f"  - 唯一重复节点数: {unique_dup_nodes}")
                avg_dup = total_dup_nodes / unique_dup_nodes if unique_dup_nodes > 0 else 0.0
                print(f"  - 平均重复度: {avg_dup:.2f}")
            if seq_parallel_world_size > 1:
                dist.barrier()
            sync_device(device)
            window_adjust_time = time.time() - window_adjust_start
            if args.rank == 0:
                print("------------------------------------------------------------------------------------")
                print("Window Adjust Time: {:.3f}s".format(window_adjust_time))
                print("------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
