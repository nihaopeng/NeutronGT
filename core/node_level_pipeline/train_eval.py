import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from core.metisPartition import weightMetis_keepParent
from gt_sp.evaluate import calc_acc
from gt_sp.initialize import get_sequence_parallel_group, get_sequence_parallel_world_size
from models.graphormer_dist_node_level import Graphormer
from models.gt_dist_node_level import GT
from models.gt_dist_node_level_single_window import GT_SW

from .runtime import sync_device


def _supports_peak_memory_metrics(device) -> bool:
    return torch.cuda.is_available() and str(device).startswith("cuda")


def _get_cuda_device_index(device) -> int:
    dev = torch.device(device)
    return torch.cuda.current_device() if dev.index is None else dev.index


def _reset_peak_memory_stats(device) -> None:
    if not _supports_peak_memory_metrics(device):
        return
    torch.cuda.reset_peak_memory_stats(_get_cuda_device_index(device))


def _get_peak_memory_mb(device):
    if not _supports_peak_memory_metrics(device):
        return 0.0, 0.0
    device_index = _get_cuda_device_index(device)
    allocated_mb = torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)
    reserved_mb = torch.cuda.max_memory_reserved(device_index) / (1024 ** 2)
    return allocated_mb, reserved_mb


def build_zero_loss(model: torch.nn.Module, device: str):
    zero_loss = torch.zeros((), device=device)
    for param in model.parameters():
        if param.requires_grad:
            zero_loss = zero_loss + param.sum() * 0.0
    return zero_loss

def build_model(args,feature,device,y,**kwargs):
    graph_in_degree = kwargs["graph_in_degree"]
    graph_out_degree = kwargs["graph_out_degree"]
    if args.model == "graphormer":
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=feature.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=y.max().item()+1,
            attn_bias_dim=args.attn_bias_dim,
            dropout_rate=args.dropout_rate,
            input_dropout_rate=args.input_dropout_rate,
            attention_dropout_rate=args.attention_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node,
            args=args,
            num_in_degree=int(torch.max(graph_in_degree).item()) if graph_in_degree is not None else 0,
            num_out_degree=int(torch.max(graph_out_degree).item()) if graph_out_degree is not None else 0,
            num_spatial=args.max_dist + 2,
            num_edges=args.max_num_edges,
            max_dist=args.max_dist,
            edge_dim=64,
        ).to(device)
    elif args.model == "gt":
        model = GT(
           n_layers=args.n_layers,
             num_heads=args.num_heads,
             input_dim=feature.shape[1],
             hidden_dim=args.hidden_dim,
             output_dim=y.max().item()+1,
             attn_bias_dim=args.attn_bias_dim,
             dropout_rate=args.dropout_rate,
             input_dropout_rate=args.input_dropout_rate,
             attention_dropout_rate=args.attention_dropout_rate,
             ffn_dim=args.ffn_dim,
             num_global_node=args.num_global_node,
             args=args,
             num_in_degree = int(torch.max(graph_in_degree).item()) if graph_in_degree is not None else 0,
             num_out_degree = int(torch.max(graph_out_degree).item()) if graph_out_degree is not None else 0,
             num_spatial=args.max_dist+2,
             num_edges=args.max_num_edges,
             max_dist=args.max_dist,
             edge_dim=64
        ).to(device)
    elif args.model == "gt_sw": # only use in ppr
        model = GT_SW(
           n_layers=args.n_layers,
             num_heads=args.num_heads,
             input_dim=feature.shape[1],
             hidden_dim=args.hidden_dim,
             output_dim=y.max().item()+1,
             attn_bias_dim=args.attn_bias_dim,
             dropout_rate=args.dropout_rate,
             input_dropout_rate=args.input_dropout_rate,
             attention_dropout_rate=args.attention_dropout_rate,
             ffn_dim=args.ffn_dim,
             num_global_node=args.num_global_node,
             args=args,
             num_in_degree = int(torch.max(graph_in_degree).item()) if graph_in_degree is not None else 0,
             num_out_degree = int(torch.max(graph_out_degree).item()) if graph_out_degree is not None else 0,
             num_spatial=args.max_dist+2,
             num_edges=args.max_num_edges,
             max_dist=args.max_dist,
             edge_dim=64
        ).to(device)
    return model

def eval_epoch(args, model, local_partition_ids, local_partitions, feature, y, split_idx, device, epoch, structInfo):
    graph_in_degree = structInfo.graph_in_degree
    graph_out_degree = structInfo.graph_out_degree
    local_spatial_pos_by_pid = structInfo.local_spatial_pos_by_pid
    local_sub_edge_index_list = structInfo.local_sub_edge_index_for_partition_results
    local_dup_indices = structInfo.local_dup_indices
    local_dup_feature_cache = structInfo.local_dup_nodes_per_partition_feature
    model.eval()
    y_train_true,y_valid_true,y_test_true = [], [], []
    y_train_pred,y_valid_pred,y_test_pred = [], [], []
    
    kv_cache_per_partition = None
    if args.use_cache:
        kv_cache_per_partition = [None] * len(local_partitions)
    
    for local_i, global_pid in enumerate(local_partition_ids):
        idx = local_partitions[local_i]
        if args.use_cache:
            dup_index_i = local_dup_indices[local_i]
            start_of_no_dup = int(dup_index_i.numel())
            x_i = feature[idx[start_of_no_dup:]].to(device)
            if start_of_no_dup > 0:
                x_i = torch.cat([local_dup_feature_cache[dup_index_i], x_i], dim=0)
        else:
            x_i = feature[idx].to(device)

        attn_bias,in_degree,out_degree = None,None,None
        edge_index_i,edge_input_i,mask,spatial_pos_i = local_sub_edge_index_list[local_i].to(device),None,None,None
        if args.struct_enc=="True":
            in_degree = graph_in_degree[idx].to(device)
            out_degree = graph_out_degree[idx].to(device)
            spatial_pos_i = local_spatial_pos_by_pid[local_i].to(device)
            assert len(idx) == spatial_pos_i.shape[0], f"rank {args.rank} spatial_pos mismatch for global_pid={global_pid}"
        current_kv_cache = kv_cache_per_partition[local_i] if kv_cache_per_partition is not None else None
        
        out_i,_,_,updated_kv_cache = model(x_i, attn_bias, edge_index_i,in_degree,out_degree, spatial_pos_i,edge_input_i,
                                          attn_type=args.attn_type,mask=mask,dup_nodes_kv_cache=current_kv_cache,part_id=global_pid)
        
        if kv_cache_per_partition is not None and updated_kv_cache is not None:
            kv_cache_per_partition[local_i] = updated_kv_cache
        mask_train = torch.isin(idx.to(device), split_idx["train"].to(device)).to('cpu')
        mask_valid = torch.isin(idx.to(device), split_idx["valid"].to(device)).to('cpu')
        mask_test = torch.isin(idx.to(device), split_idx["test"].to(device)).to('cpu')
        y_train_true.append(y[idx][mask_train])
        y_valid_true.append(y[idx][mask_valid])
        y_test_true.append(y[idx][mask_test])
        y_train_pred.append(out_i.argmax(1)[mask_train])
        y_valid_pred.append(out_i.argmax(1)[mask_valid])
        y_test_pred.append(out_i.argmax(1)[mask_test])

    local_eval = {
        "train_true": torch.cat(y_train_true).cpu() if y_train_true else torch.empty(0, dtype=y.dtype),
        "valid_true": torch.cat(y_valid_true).cpu() if y_valid_true else torch.empty(0, dtype=y.dtype),
        "test_true": torch.cat(y_test_true).cpu() if y_test_true else torch.empty(0, dtype=y.dtype),
        "train_pred": torch.cat(y_train_pred).cpu() if y_train_pred else torch.empty(0, dtype=torch.long),
        "valid_pred": torch.cat(y_valid_pred).cpu() if y_valid_pred else torch.empty(0, dtype=torch.long),
        "test_pred": torch.cat(y_test_pred).cpu() if y_test_pred else torch.empty(0, dtype=torch.long),
    }
    gathered = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered, local_eval)

    if args.rank == 0:
        y_train_true = torch.cat([item["train_true"] for item in gathered if item["train_true"].numel() > 0]) if any(item["train_true"].numel() > 0 for item in gathered) else torch.empty(0, dtype=y.dtype)
        y_valid_true = torch.cat([item["valid_true"] for item in gathered if item["valid_true"].numel() > 0]) if any(item["valid_true"].numel() > 0 for item in gathered) else torch.empty(0, dtype=y.dtype)
        y_test_true = torch.cat([item["test_true"] for item in gathered if item["test_true"].numel() > 0]) if any(item["test_true"].numel() > 0 for item in gathered) else torch.empty(0, dtype=y.dtype)
        y_train_pred = torch.cat([item["train_pred"] for item in gathered if item["train_pred"].numel() > 0]) if any(item["train_pred"].numel() > 0 for item in gathered) else torch.empty(0, dtype=torch.long)
        y_valid_pred = torch.cat([item["valid_pred"] for item in gathered if item["valid_pred"].numel() > 0]) if any(item["valid_pred"].numel() > 0 for item in gathered) else torch.empty(0, dtype=torch.long)
        y_test_pred = torch.cat([item["test_pred"] for item in gathered if item["test_pred"].numel() > 0]) if any(item["test_pred"].numel() > 0 for item in gathered) else torch.empty(0, dtype=torch.long)
        train_acc = calc_acc(y_train_true,y_train_pred) if y_train_true.numel() > 0 else 0.0
        valid_acc = calc_acc(y_valid_true,y_valid_pred) if y_valid_true.numel() > 0 else 0.0
        test_acc = calc_acc(y_test_true,y_test_pred) if y_test_true.numel() > 0 else 0.0
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Epoch: {:03d}, train_acc: {:.5f}%, valid_acc: {:.5f}%, test_acc: {:.5f}%, samples(train/valid/test)=({}/{}/{})".format(
            epoch, train_acc*100, valid_acc*100, test_acc*100,
            y_train_true.numel(), y_valid_true.numel(), y_test_true.numel()))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return train_acc*100, valid_acc*100, test_acc*100, kv_cache_per_partition
    return None, None, None, kv_cache_per_partition

def train_epoch(args, model:torch.nn.Module, local_partition_ids, local_partitions, feature, y, optimizer, lr_scheduler, world_size, split_idx, device, epoch, structInfo):
    graph_in_degree = structInfo.graph_in_degree
    graph_out_degree = structInfo.graph_out_degree
    local_spatial_pos_by_pid = structInfo.local_spatial_pos_by_pid
    local_sub_edge_index_list = structInfo.local_sub_edge_index_for_partition_results
    local_dup_indices = structInfo.local_dup_indices
    local_dup_feature_cache = structInfo.local_dup_nodes_per_partition_feature
    model.train()
    loss_list = []
    scores_by_pid = {}
    cpu_to_gpu_total_time = 0.0
    window_forward_backward_total_time = 0.0
    window_count = 0

    local_window_count = len(local_partitions)
    max_window_steps = local_window_count
    if world_size > 1:
        local_window_count_tensor = torch.tensor([local_window_count], device=device, dtype=torch.long)
        gathered_window_counts = [torch.zeros_like(local_window_count_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_window_counts, local_window_count_tensor)
        max_window_steps = max(int(item.item()) for item in gathered_window_counts)

    sync_device(device)
    _reset_peak_memory_stats(device)
    epoch_train_start = time.time()

    kv_cache_per_partition = None
    if args.use_cache:
        kv_cache_per_partition = [None] * len(local_partitions)

    for local_i in range(max_window_steps):
        is_dummy_step = local_i >= local_window_count
        if not is_dummy_step:
            global_pid = local_partition_ids[local_i]
            idx_i = local_partitions[local_i]

            sync_device(device)
            cpu_to_gpu_start = time.time()
            if args.use_cache:
                dup_index_i = local_dup_indices[local_i]
                start_of_no_dup = int(dup_index_i.numel())
                x_i = feature[idx_i[start_of_no_dup:]].to(device)
                if start_of_no_dup > 0:
                    x_i = torch.cat([local_dup_feature_cache[dup_index_i], x_i], dim=0)
            else:
                x_i = feature[idx_i].to(device)
            attn_bias, in_degree, out_degree = None, None, None
            edge_index_i, edge_input_i, mask, spatial_pos_i = local_sub_edge_index_list[local_i].to(device), None, None, None
            if args.struct_enc == "True":
                in_degree = graph_in_degree[idx_i].to(device)
                out_degree = graph_out_degree[idx_i].to(device)
                spatial_pos_i = local_spatial_pos_by_pid[local_i].to(device)
                assert len(idx_i) == spatial_pos_i.shape[0], f"rank {args.rank} spatial_pos mismatch for global_pid={global_pid}"
            sync_device(device)
            cpu_to_gpu_total_time += time.time() - cpu_to_gpu_start

            current_kv_cache = kv_cache_per_partition[local_i] if kv_cache_per_partition is not None else None

            sync_device(device)
            window_train_start = time.time()
            out_i, score_agg, score_spe, updated_kv_cache = model(
                x_i,
                attn_bias,
                edge_index_i,
                in_degree,
                out_degree,
                spatial_pos_i,
                edge_input_i,
                attn_type=args.attn_type,
                mask=mask,
                dup_nodes_kv_cache=current_kv_cache,
                part_id=global_pid
            )

            if kv_cache_per_partition is not None and updated_kv_cache is not None:
                kv_cache_per_partition[local_i] = updated_kv_cache

            scores_by_pid[global_pid] = score_spe[args.n_layers - 1]
            loss = F.nll_loss(out_i, y[idx_i].to(device).long(), reduction='none')
            mask_train = torch.isin(idx_i.to(device), split_idx["train"].to(device))
            if mask_train.any():
                loss = loss[mask_train].mean()
                loss_list.append(loss.item())
            else:
                print(f"rank:{args.rank},epoch:{epoch},no train nodes!")
                loss = build_zero_loss(model, device)
                loss_list.append(0.0)
        else:
            loss = build_zero_loss(model, device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if world_size > 1:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad.div_(get_sequence_parallel_world_size())
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=get_sequence_parallel_group())
        optimizer.step()

        if not is_dummy_step:
            sync_device(device)
            window_forward_backward_total_time += time.time() - window_train_start
            window_count += 1

    lr_scheduler.step()
    sync_device(device)
    epoch_train_total_time = time.time() - epoch_train_start
    local_peak_allocated_mb, local_peak_reserved_mb = _get_peak_memory_mb(device)
    max_peak_allocated_mb = local_peak_allocated_mb
    max_peak_reserved_mb = local_peak_reserved_mb
    if world_size > 1:
        peak_tensor = torch.tensor([local_peak_allocated_mb, local_peak_reserved_mb], device=device, dtype=torch.float64)
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        max_peak_allocated_mb = float(peak_tensor[0].item())
        max_peak_reserved_mb = float(peak_tensor[1].item())

    loss_mean = float(np.mean(loss_list)) if loss_list else 0.0
    time_stats = {
        "epoch_train_total_time": epoch_train_total_time,
        "cpu_to_gpu_total_time": cpu_to_gpu_total_time,
        "window_forward_backward_total_time": window_forward_backward_total_time,
        "window_forward_backward_avg_time": window_forward_backward_total_time / window_count if window_count > 0 else 0.0,
        "num_processed_windows": window_count,
        "local_peak_allocated_mb": local_peak_allocated_mb,
        "local_peak_reserved_mb": local_peak_reserved_mb,
        "max_peak_allocated_mb": max_peak_allocated_mb,
        "max_peak_reserved_mb": max_peak_reserved_mb,
    }
    print(
        "Peak GPU memory: rank={}, allocated_mb={:.2f}, reserved_mb={:.2f}".format(
            args.rank,
            local_peak_allocated_mb,
            local_peak_reserved_mb,
        )
    )
    if args.rank == 0:
        print("------------------------------------------------------------------------------------")
        print(
            "Epoch: {:03d}, Loss: {:.4f}, Train Time: {:.3f}s, CPU->GPU Time: {:.3f}s, Window FW/BW Avg: {:.3f}s, Window FW/BW Total: {:.3f}s, Windows: {}, Max Window Steps: {}".format(
                epoch,
                loss_mean,
                time_stats["epoch_train_total_time"],
                time_stats["cpu_to_gpu_total_time"],
                time_stats["window_forward_backward_avg_time"],
                time_stats["window_forward_backward_total_time"],
                time_stats["num_processed_windows"],
                max_window_steps,
            )
        )
        print("Training epoch time: {:.3f}s".format(time_stats["epoch_train_total_time"]))
        print(
            "Peak GPU memory summary: max_reserved_mb={:.2f}, max_allocated_mb={:.2f}".format(
                time_stats["max_peak_reserved_mb"],
                time_stats["max_peak_allocated_mb"],
            )
        )
        print("------------------------------------------------------------------------------------")
    return loss_mean, scores_by_pid, kv_cache_per_partition, time_stats
