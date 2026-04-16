import argparse
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist

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
    build_checkpoint_payload,
    build_graph_struct_info,
    build_local_partitions,
    build_model,
    ensure_checkpoint_dir,
    eval_epoch,
    load_optional_edge_csr,
    load_training_checkpoint,
    restore_global_window_state,
    save_training_checkpoint,
    sync_device,
    train_epoch,
    validate_resume_supported,
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
    
    if args.resume_checkpoint or args.resume_latest:
        validate_resume_supported(args)

    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        if args.save_checkpoint or args.resume_latest:
            ensure_checkpoint_dir(args)
    
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
    wm:weightMetis_keepParent = structInfo.wm

    broadcast_window_state(args, structInfo, feature, device)
    sync_device(device)

    local_partition_ids, local_partitions = build_local_partitions(structInfo, args.rank, seq_parallel_world_size)
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

    resume_state = load_training_checkpoint(args, model, optimizer, lr_scheduler, device)
    start_epoch = resume_state.start_epoch
    loss_mean_list = resume_state.loss_mean_list or []
    best_val = resume_state.best_val
    best_test = resume_state.best_test
    if args.rank == 0:
        if resume_state.checkpoint_path:
            print(f"Resumed training from: {resume_state.checkpoint_path}")
            print(f"Resume start epoch: {start_epoch}")
            print(f"Checkpoint best metrics: best_val={best_val:.5f}, best_test={best_test:.5f}")
        elif resume_state.resume_requested:
            ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else args.model_dir
            print(f"No checkpoint found under {ckpt_dir}, starting training from scratch.")

    detector = LossStagnationDetector(cooldown=0)
    
    for epoch in range(start_epoch, args.epochs):
        loss_mean,scores_by_pid,updated_kv_cache,time_stats = train_epoch(args,model,local_partition_ids,local_partitions,feature,y,optimizer,lr_scheduler,seq_parallel_world_size,split_idx,device,
                    epoch=epoch,
                    structInfo=structInfo)
        
        is_best_checkpoint = False
        if epoch % 20 == 0:
            if args.rank == 0:
                print(f"epoch {epoch}: lr = {lr_scheduler.get_last_lr()[0]:.2e}")
            train_acc, valid_acc, test_acc, _ = eval_epoch(args,model,local_partition_ids,local_partitions,feature,y,split_idx,device,
                    epoch=epoch,
                    structInfo=structInfo)
            if args.rank == 0 and valid_acc is not None:
                if valid_acc > best_val:
                    best_val = valid_acc
                    best_test = test_acc if test_acc is not None else best_test
                    is_best_checkpoint = True
                    if args.save_model:
                        torch.save(model.state_dict(), os.path.join(args.model_dir, f'{args.dataset}.pkl'))
                elif test_acc is not None and test_acc > best_test:
                    best_test = test_acc
        
        loss_mean_list.append(loss_mean)
        if args.use_cache==0 and detector(loss_mean_list):
            if seq_parallel_world_size > 1:
                dist.barrier()
            sync_device(device)
            window_adjust_start = time.time()
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
            if seq_parallel_world_size > 1:
                dist.barrier()
            sync_device(device)
            _ = time.time() - window_adjust_start

        if args.save_checkpoint:
            if args.rank == 0:
                payload = build_checkpoint_payload(
                    args,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    loss_mean_list,
                    structInfo.window_state_version,
                    best_val,
                    best_test,
                )
                saved_paths = save_training_checkpoint(args, payload, epoch, is_best=is_best_checkpoint)
                print(f"Saved checkpoint(s): {', '.join(saved_paths)}")
            if seq_parallel_world_size > 1:
                dist.barrier()


if __name__ == "__main__":
    main()
