import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from core.metisPartition import weightMetis_keepParent
from models.graphormer_dist_node_level import Graphormer
from models.gt_dist_node_level import GT
from models.gt_dist_node_level_single_window import GT_SW
from utils.lr import PolynomialDecayLR
import argparse
import os
import time
import random
import pandas as pd
import torch.distributed as dist
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    set_last_batch_global_token_indices,
)
from gt_sp.reducer import sync_params_and_buffers, Reducer
from gt_sp.evaluate import calc_acc
from gt_sp.utils import LossStagnationDetector, compute_graphormer_spatial_pos_only, get_node_degrees, random_split_idx
from utils.parser_node_level import parser_add_main_args
from core.pprPartition import add_isolated_connections, personal_pagerank,build_adj_fromat
from utils.vis import vis_interface
import utils.vis as vis
import utils.logger as logger

def sync_device(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)

# to avoid the lock when every rank get different numbers of partition
def build_zero_loss(model: torch.nn.Module, device: str):
    zero_loss = torch.zeros((), device=device)
    for param in model.parameters():
        if param.requires_grad:
            zero_loss = zero_loss + param.sum() * 0.0
    return zero_loss


class StructInfo:
    def __init__(self,**kwargs) -> None:
        self.graph_in_degree = kwargs["graph_in_degree"]
        self.graph_out_degree = kwargs["graph_out_degree"]
        self.sorted_ppr_matrix = kwargs["sorted_ppr_matrix"]  # tuple[torch.Tensor, torch.Tensor]
        self.wm = kwargs["wm"]
        self.spatial_pos_by_pid = None
        self.sub_edge_index_list = None


def build_local_partitions(wm: weightMetis_keepParent, rank: int, world_size: int):
    local_partition_ids = list(range(rank, len(wm.partitioned_results), world_size))
    local_partitions = [wm.partitioned_results[pid] for pid in local_partition_ids]
    return local_partition_ids, local_partitions


def build_dup_cache_metadata(wm: weightMetis_keepParent, feature: torch.Tensor, device: str):
    all_dup_nodes = torch.cat(wm.dup_nodes_per_partition)
    dup_unique_sorted = torch.unique(all_dup_nodes, sorted=True).flip(0)
    hash_index = {int(node): i for i, node in enumerate(dup_unique_sorted)}
    wm.dup_indices = [
        [hash_index[int(node)] for node in partition]
        for partition in wm.dup_nodes_per_partition
    ]
    wm.dup_nodes_per_partition_feature = feature[torch.tensor(dup_unique_sorted)].to(device)
    return dup_unique_sorted


def broadcast_window_state(args, structInfo: StructInfo, feature: torch.Tensor, device: str):
    if args.world_size <= 1:
        if args.use_cache:
            build_dup_cache_metadata(structInfo.wm, feature, device)
        return

    payload = [None]
    if args.rank == 0:
        payload[0] = {
            "partitioned_results": structInfo.wm.partitioned_results,
            "sub_edge_index_for_partition_results": structInfo.wm.sub_edge_index_for_partition_results,
            "dup_nodes_per_partition": structInfo.wm.dup_nodes_per_partition,
            "spatial_pos_by_pid": structInfo.spatial_pos_by_pid,
        }
    dist.broadcast_object_list(payload, src=0)
    state = payload[0]
    if args.rank != 0:
        structInfo.wm.partitioned_results = state["partitioned_results"]
        structInfo.wm.sub_edge_index_for_partition_results = state["sub_edge_index_for_partition_results"]
        structInfo.wm.dup_nodes_per_partition = state["dup_nodes_per_partition"]
        structInfo.spatial_pos_by_pid = state["spatial_pos_by_pid"]

    if args.use_cache:
        build_dup_cache_metadata(structInfo.wm, feature, device)

def build_graph_struct_info(args,N,edge_index,feature,world_size,device,topk=50,n_parts=50,related_nodes_topk_rate=5,connect_prob=0.01):
    # --------------------计算结构信息------------------------------------------------------------
    # =================== ppr partition =========================
    # partitioned_results = []
    # if args.rank == 0:
    sorted_ppr_matrix = personal_pagerank(
        edge_index,
        args.ppr_alpha,
        topk=topk,
        backend=args.ppr_backend,
        num_iterations=args.ppr_num_iterations,
        batch_size=args.ppr_batch_size,
        eps=args.ppr_eps,
        device=device,
    )
    sorted_ppr_matrix = add_isolated_connections(sorted_ppr_matrix, edge_index, N, connect_prob=connect_prob)
    csr_adjacency,eweights,adj_weight = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)
    wm = weightMetis_keepParent(
        csr_adjacency=csr_adjacency, 
        eweights=eweights,
        n_parts=n_parts,
        feature=feature,
        edge_index=edge_index,
        related_nodes_topk_rate=related_nodes_topk_rate,
        attn_type=args.attn_type,
        sorted_ppr_matrix=sorted_ppr_matrix)
    
    # ===== 计算centrality encoding =====
    graph_in_degree, graph_out_degree = None, None
    if args.struct_enc == "True":
        graph_in_degree, graph_out_degree = get_node_degrees(edge_index, N)

    print("node len:",end="")
    sum_nodes_in_compute = 0 
    for p in wm.partitioned_results:
        print(len(p),end="|")
        sum_nodes_in_compute += len(p)
    print("\nedge len:",end="")
    sum_edges_in_compute = 0 
    for p in wm.sub_edge_index_for_partition_results:
        print(len(p[0]),end="|")
        sum_edges_in_compute += len(p[0])
    print(f"\nsum nodes in compute:{sum_nodes_in_compute},sum edges in compute:{sum_edges_in_compute}")
    
    return StructInfo(
        graph_in_degree=graph_in_degree,
        graph_out_degree=graph_out_degree,
        sorted_ppr_matrix=sorted_ppr_matrix,
        wm=wm)

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

@torch.no_grad()
def eval_epoch(args, model, local_partition_ids, local_partitions, feature, y, split_idx, device, epoch, structInfo):
    graph_in_degree = structInfo.graph_in_degree
    graph_out_degree = structInfo.graph_out_degree
    spatial_pos_by_pid = structInfo.spatial_pos_by_pid
    wm:weightMetis_keepParent = structInfo.wm
    model.eval()
    y_train_true,y_valid_true,y_test_true = [], [], []
    y_train_pred,y_valid_pred,y_test_pred = [], [], []
    
    kv_cache_per_partition = None
    if args.use_cache:
        kv_cache_per_partition = [None] * len(local_partitions)
    
    for local_i, global_pid in enumerate(local_partition_ids):
        idx = local_partitions[local_i]
        assert torch.equal(idx, wm.partitioned_results[global_pid]), f"rank {args.rank} partition mismatch for global_pid={global_pid}"
        if args.use_cache:
            start_of_no_dup = len(wm.dup_indices[global_pid])
            x_i = feature[idx[start_of_no_dup:]].to(device)
            x_i = torch.cat([wm.dup_nodes_per_partition_feature[wm.dup_indices[global_pid]],x_i],dim=0)
        else:
            x_i = feature[idx].to(device)

        attn_bias,in_degree,out_degree = None,None,None
        edge_index_i,edge_input_i,mask,spatial_pos_i = wm.sub_edge_index_for_partition_results[global_pid].to(device),None,None,None
        if args.struct_enc=="True":
            in_degree = graph_in_degree[idx].to(device)
            out_degree = graph_out_degree[idx].to(device)
            spatial_pos_i = spatial_pos_by_pid[global_pid].to(device)
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
    spatial_pos_by_pid = structInfo.spatial_pos_by_pid
    wm:weightMetis_keepParent = structInfo.wm
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
    epoch_train_start = time.time()

    kv_cache_per_partition = None
    if args.use_cache:
        kv_cache_per_partition = [None] * len(local_partitions)

    for local_i in range(max_window_steps):
        is_dummy_step = local_i >= local_window_count
        if not is_dummy_step:
            global_pid = local_partition_ids[local_i]
            idx_i = local_partitions[local_i]
            assert torch.equal(idx_i, wm.partitioned_results[global_pid]), f"rank {args.rank} partition mismatch for global_pid={global_pid}"

            sync_device(device)
            cpu_to_gpu_start = time.time()
            if args.use_cache:
                start_of_no_dup = len(wm.dup_indices[global_pid])
                x_i = feature[idx_i[start_of_no_dup:]].to(device)
                x_i = torch.cat([wm.dup_nodes_per_partition_feature[wm.dup_indices[global_pid]], x_i], dim=0)
            else:
                x_i = feature[idx_i].to(device)
            attn_bias, in_degree, out_degree = None, None, None
            edge_index_i, edge_input_i, mask, spatial_pos_i = wm.sub_edge_index_for_partition_results[global_pid].to(device), None, None, None
            if args.struct_enc == "True":
                in_degree = graph_in_degree[idx_i].to(device)
                out_degree = graph_out_degree[idx_i].to(device)
                spatial_pos_i = spatial_pos_by_pid[global_pid].to(device)
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
    loss_mean = float(np.mean(loss_list)) if loss_list else 0.0
    time_stats = {
        "epoch_train_total_time": epoch_train_total_time,
        "cpu_to_gpu_total_time": cpu_to_gpu_total_time,
        "window_forward_backward_total_time": window_forward_backward_total_time,
        "window_forward_backward_avg_time": window_forward_backward_total_time / window_count if window_count > 0 else 0.0,
        "num_processed_windows": window_count,
    }
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
        print("------------------------------------------------------------------------------------")
    return loss_mean, scores_by_pid, kv_cache_per_partition, time_stats

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
    edge_index = torch.load(args.dataset_dir + args.dataset + '/edge_index.pt')
    N = feature.shape[0]

    if args.dataset == 'pokec':
        y = torch.clamp(y, min=0) 
    split_idx = random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)

    if args.rank == 0:
        print(args)
        print('Dataset load successfully')
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
            n_parts=10,
            related_nodes_topk_rate=2
        )
    wm:weightMetis_keepParent = structInfo.wm

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

    ppr_tuple = (structInfo.sorted_ppr_matrix[0], structInfo.sorted_ppr_matrix[1])
    spatial_pos_preprocess_time = 0.0
    if args.struct_enc=="True":
        sync_device(device)
        spatial_pos_preprocess_start = time.time()
        structInfo.spatial_pos_by_pid,_ = compute_graphormer_spatial_pos_only(ppr_tuple, wm.partitioned_results, N, max_dist=args.max_dist)
        sync_device(device)
        spatial_pos_preprocess_time = time.time() - spatial_pos_preprocess_start
    sync_device(device)
    graph_preprocess_total_time = time.time() - graph_preprocess_start
    if args.rank == 0:
        print("====================================================================================")
        print("Graph preprocess total time: {:.3f}s".format(graph_preprocess_total_time))
        if args.struct_enc == "True":
            print("Spatial position preprocess time: {:.3f}s".format(spatial_pos_preprocess_time))
        print("====================================================================================")

    if args.use_cache:
        dup_unique_sorted = build_dup_cache_metadata(wm, feature, device)
        if args.rank == 0:
            print("\n" + "="*80)
            print("重复节点统计信息:")
            print("="*80)
            total_dup_nodes = len(dup_unique_sorted)
            total_nodes = N
            dup_ratio_global = total_dup_nodes / total_nodes * 100
            print(f"全局统计:")
            print(f"  - 总节点数: {total_nodes}")
            print(f"  - 重复节点数: {total_dup_nodes}")
            print(f"  - 重复节点占比: {dup_ratio_global:.2f}%")
            print(f"\n各分区统计 (前10个分区):")
            for i, (partition, dup_nodes) in enumerate(zip(wm.partitioned_results, wm.dup_nodes_per_partition)):
                if i >= 10:
                    break 
                partition_size = len(partition)
                dup_size = len(dup_nodes)
                dup_ratio = dup_size / partition_size * 100 if partition_size > 0 else 0
                print(f"  分区 {i:2d}: 总节点={partition_size:4d}, 重复节点={dup_size:4d}, 占比={dup_ratio:6.2f}%")
            if len(wm.partitioned_results) > 10:
                print(f"  ... 还有 {len(wm.partitioned_results)-10} 个分区未显示")

    broadcast_window_state(args, structInfo, feature, device)
    local_partition_ids, local_partitions = build_local_partitions(wm, args.rank, seq_parallel_world_size)
    print(f"rank {args.rank} local_partition_ids: {local_partition_ids}")
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
                merged_scores_by_pid = {}
                for item in gathered_scores:
                    merged_scores_by_pid.update(item)
                ordered_scores = [merged_scores_by_pid[pid] for pid in range(len(wm.partitioned_results))]
                wm.node_out(ordered_scores,remove_ratio=0.3)
                wm.node_in()
                if args.struct_enc=="True":
                    structInfo.spatial_pos_by_pid,_ = compute_graphormer_spatial_pos_only(ppr_tuple, wm.partitioned_results, N, max_dist=args.max_dist)
            broadcast_window_state(args, structInfo, feature, device)
            local_partition_ids, local_partitions = build_local_partitions(wm, args.rank, seq_parallel_world_size)
            if args.rank == 0 and hasattr(wm, 'dup_nodes_per_partition'):
                print(f"\n[窗口调整后] 重复节点统计:")
                total_dup_nodes = sum(len(dup_nodes) for dup_nodes in wm.dup_nodes_per_partition)
                unique_dup_nodes = len(torch.unique(torch.cat(wm.dup_nodes_per_partition)))
                total_partition_nodes = sum(len(partition) for partition in wm.partitioned_results)
                print(f"  - 总分区节点数: {total_partition_nodes}")
                print(f"  - 重复节点出现次数: {total_dup_nodes}")
                print(f"  - 唯一重复节点数: {unique_dup_nodes}")
                print(f"  - 平均重复度: {total_dup_nodes/unique_dup_nodes:.2f}")
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
