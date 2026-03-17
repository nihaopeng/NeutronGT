import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from utils.lr import PolynomialDecayLR
import argparse
import math
from tqdm import tqdm
import scipy.sparse as sp
import copy
import os
import time
import random
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch_geometric.utils import degree
from gt_sp.evaluate import calc_acc
from gt_sp.initialize import (
    initialize_distributed,
    initialize_sequence_parallel,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    set_last_batch_global_token_indices,
    get_last_batch_flag,
    last_batch_flag,
)
from gt_sp.reducer import sync_params_and_buffers, Reducer
from gt_sp.utils import pad_x_bs, pad_2d_bs
from data.dataset import GraphormerDataset
from models.gt_dist_graph_level_mp_malnet_ppr import GT
from models.graphormer_dist_graph_level_mp_malnet_ppr import Graphormer
from core.pprPartition import personal_pagerank, add_isolated_connections, build_adj_fromat
from core.metisPartition import weightMetis_keepParent
from utils.parser_graph_level import parser_add_main_args
import torch.multiprocessing as mp
from datetime import timedelta


reducer = Reducer()

# ==============================================================================
# 超参数配置 - 在此调整分区数量和其他参数
# ==============================================================================
# 分区数量: 将每个图分成多少个分区
# 注意: 分区数量应该小于节点数量，建议值为 min(50, max(2, num_nodes // 100))
N_PARTS = 5  # <--- 最大分区数量

# PPR 参数
PPR_ALPHA = 0.85  # PPR 阻尼系数
PPR_TOPK = 50  # 每个节点的 PPR 邻居数量
CONNECT_PROB = 0.01  # 孤立节点连接概率

# METIS 参数
RELATED_NODES_TOPK_RATE = 5  # 关联节点扩展比例
# ==============================================================================


def get_adaptive_n_parts(num_nodes, max_n_parts=N_PARTS):
    """
    根据节点数量动态调整分区数量
    
    Args:
        num_nodes: 图的节点数量
        max_n_parts: 最大分区数量
    
    Returns:
        合适的分区数量
    """
    # 每个分区至少要有一些节点，这里设置每个分区平均至少 50 个节点
    min_nodes_per_part = 50
    calculated_parts = max(2, num_nodes // min_nodes_per_part)
    return min(calculated_parts, max_n_parts)


def reduce_hook(param, name, grad):
    reducer.reduce(param, name, grad)


class GraphStructInfo:
    """存储图的结构信息，包括分区结果和边信息"""
    def __init__(self, wm, in_degree, out_degree):
        self.wm = wm
        self.in_degree = in_degree
        self.out_degree = out_degree


def split_batch(batch):
    """
    将 batch 拆分成多个独立的图
    
    Args:
        batch: 包含多个图的 batch
    
    Returns:
        graphs: 列表，每个元素是 (x, y, in_degree, out_degree, edge_index, num_nodes) 元组
    """
    bs = batch.graph_node_num.shape[0]  # batch size
    graph_node_num = batch.graph_node_num.tolist()  # 每个图的节点数
    
    graphs = []
    node_offset = 0
    
    for i in range(bs):
        num_nodes = graph_node_num[i]
        
        # 提取第 i 个图的特征
        x_i = batch.x[i, :num_nodes, :]  # [num_nodes, feature_dim]
        
        # 提取边（需要根据节点偏移量筛选和重映射）
        # 边索引是扁平化的，需要筛选出属于当前图的边
        edge_index_global = batch.edge_index  # [2, total_edges]
        
        # 找到属于当前图的边：边的两端都在 [node_offset, node_offset + num_nodes) 范围内
        src, dst = edge_index_global[0], edge_index_global[1]
        mask = (src >= node_offset) & (src < node_offset + num_nodes) & \
               (dst >= node_offset) & (dst < node_offset + num_nodes)
        
        # 重映射边的节点ID（减去偏移量）
        edge_index_local = torch.stack([
            src[mask] - node_offset,
            dst[mask] - node_offset
        ], dim=0)
        
        # 标签
        y_i = batch.y[i]
        
        # 度数（已经是原始的，不需要重映射）
        in_degree_i = batch.in_degree[i, :num_nodes]
        out_degree_i = batch.out_degree[i, :num_nodes]
        
        graphs.append((x_i, y_i, in_degree_i, out_degree_i, edge_index_local, num_nodes))
        
        # 更新偏移量
        node_offset += num_nodes
    
    return graphs


def build_graph_struct_info(args, x, edge_index, num_nodes):
    """
    对单个图进行 PPR + METIS 分区
    
    Args:
        args: 命令行参数
        x: 节点特征 [num_nodes, feature_dim]
        edge_index: 边索引 [2, num_edges]
        num_nodes: 节点数量
    
    Returns:
        GraphStructInfo 对象
    """
    if args.rank == 0:
        print(f"Building graph structure info for graph with {num_nodes} nodes...")
    
    # 动态调整分区数量
    n_parts = get_adaptive_n_parts(num_nodes)
    if args.rank == 0 and n_parts < N_PARTS:
        print(f"Adapting N_PARTS from {N_PARTS} to {n_parts} for small graph")
    
    # 验证数据一致性
    assert x.shape[0] == num_nodes, f"Feature mismatch: x has {x.shape[0]} rows but expected {num_nodes} nodes"
    if edge_index.numel() > 0:
        assert edge_index.max().item() < num_nodes, f"Edge index contains invalid node IDs"

    # 处理空边图（没有边的图）
    if edge_index.numel() == 0 or edge_index.shape[1] == 0:
        if args.rank == 0:
            print(f"Warning: Graph has no edges, creating dummy structure")
        # 为空图创建简单的度数
        in_degree = torch.zeros(num_nodes, dtype=torch.long)
        out_degree = torch.zeros(num_nodes, dtype=torch.long)
        
        # 返回空的分区信息
        class DummyWM:
            partitioned_results = [torch.arange(num_nodes, dtype=torch.long)]
            sub_edge_index_for_partition_results = [torch.zeros((2, 0), dtype=torch.long)]
            dup_nodes_per_partition = [torch.zeros(0, dtype=torch.long)]
        
        return GraphStructInfo(DummyWM(), in_degree, out_degree)

    # 1. 计算 PPR
    sorted_ppr_matrix = personal_pagerank(edge_index, PPR_ALPHA, topk=PPR_TOPK)
    sorted_ppr_matrix = add_isolated_connections(sorted_ppr_matrix, num_nodes, connect_prob=CONNECT_PROB)
    
    # 2. 构建 CSR 邻接矩阵
    csr_adjacency, eweights, adj_weight = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)
    
    # 3. METIS 分区
    wm = weightMetis_keepParent(
        csr_adjacency=csr_adjacency,
        eweights=eweights,
        feature=x,
        edge_index=edge_index,
        n_parts=n_parts,  # 使用动态调整的分区别
        related_nodes_topk_rate=RELATED_NODES_TOPK_RATE,
        attn_type=args.attn_type,
        sorted_ppr_matrix=sorted_ppr_matrix
    )
    
    # 4. 计算节点度数
    in_degree = degree(edge_index[1], num_nodes)
    out_degree = degree(edge_index[0], num_nodes)
    
    if args.rank == 0:
        print(f"Partition results: {len(wm.partitioned_results)} partitions")
        total_nodes = sum(len(p) for p in wm.partitioned_results)
        print(f"Total nodes in partitions: {total_nodes}")
    
    return GraphStructInfo(wm, in_degree, out_degree)


def get_partition_data(args, batch, device, struct_info, partition_idx):
    """
    获取指定分区的数据
    
    Args:
        args: 命令行参数
        batch: 包含 (x, y, in_degree, out_degree, edge_index, sub_split_seq_lens) 的元组
        device: 计算设备
        struct_info: GraphStructInfo 对象
        partition_idx: 分区索引
    
    Returns:
        分区的特征、标签、度数、边索引
    """
    x_full, y_full, in_degree_full, out_degree_full, edge_index_full, _ = batch
    
    wm = struct_info.wm
    
    # 获取当前分区的节点索引
    partition_nodes = wm.partitioned_results[partition_idx]  # 全局节点索引
    sub_edge_index = wm.sub_edge_index_for_partition_results[partition_idx]
    
    # 获取重复节点信息（用于 KV cache）
    dup_indices = wm.dup_nodes_per_partition[partition_idx]
    
    # 处理节点特征
    if len(dup_indices) > 0:
        # 有重复节点，需要分离
        dup_nodes = dup_indices
        non_dup_nodes = partition_nodes[partition_nodes != dup_nodes[:, None]].squeeze()
        
        # 提取特征
        dup_features = wm.dup_nodes_per_partition_feature[partition_idx]
        if len(non_dup_nodes) > 0:
            non_dup_features = x_full[0, non_dup_nodes, :]
            x_partition = torch.cat([dup_features, non_dup_features], dim=0)
        else:
            x_partition = dup_features
    else:
        x_partition = x_full[0, partition_nodes, :]
    
    # 获取度数
    in_degree_partition = struct_info.in_degree[partition_nodes].to(device)
    out_degree_partition = struct_info.out_degree[partition_nodes].to(device)
    
    # 获取边索引并调整
    edge_index_partition = sub_edge_index.to(device)
    
    # 标签（所有分区使用相同的标签）
    y_partition = y_full.to(device)
    
    return x_partition, y_partition, in_degree_partition, out_degree_partition, edge_index_partition, dup_indices


def train(args, model, device, packed_data, struct_info_list, optimizer, criterion, epoch, lr_scheduler):
    """
    训练函数 - 按分区顺序处理
    
    Args:
        args: 命令行参数
        model: GT 模型
        device: 计算设备
        packed_data: 打包后的数据列表 [(x, y, in_degree, out_degree, edge_index, num_nodes), ...]
        struct_info_list: 每个图的 GraphStructInfo 列表
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前 epoch
        lr_scheduler: 学习率调度器
    """
    model.train()
    model.to(device)

    loss_list, iter_t_list = [], []
    
    # 遍历每个图
    for graph_idx, (x, y, in_degree, out_degree, edge_index, num_nodes) in enumerate(packed_data):
        struct_info = struct_info_list[graph_idx]
        wm = struct_info.wm
        
        # 初始化 KV cache
        num_partitions = len(wm.partitioned_results)
        kv_cache_per_partition = [None] * num_partitions
        
        # 按分区顺序处理
        for part_idx in range(num_partitions):
            # 获取当前分区数据
            partition_nodes = wm.partitioned_results[part_idx]
            sub_edge_index = wm.sub_edge_index_for_partition_results[part_idx]
            dup_indices = wm.dup_nodes_per_partition[part_idx]
            
            # 准备特征
            x_i = x[partition_nodes, :].to(device)
            
            # 准备度数
            in_degree_i = struct_info.in_degree[partition_nodes].to(device)
            out_degree_i = struct_info.out_degree[partition_nodes].to(device)
            
            # 准备边索引
            edge_index_i = sub_edge_index.to(device)
            
            # 获取当前分区的 KV cache
            current_kv_cache = kv_cache_per_partition[part_idx]
            
            t0 = time.time()
            
            # 前向传播
            pred, updated_kv_cache = model(
                x_i.unsqueeze(0),  # 添加 batch 维度
                in_degree_i.unsqueeze(0),
                out_degree_i.unsqueeze(0),
                edge_index_i,
                attn_type=args.attn_type,
                dup_nodes_kv_cache=current_kv_cache,
                part_id=part_idx
            )
            
            # 更新 KV cache
            if updated_kv_cache is not None:
                kv_cache_per_partition[part_idx] = updated_kv_cache
            
            # 计算损失
            y_true = y.to(device)
            loss = criterion(pred.squeeze(0), y_true)
            
            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # 梯度同步
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param.grad.div_(get_sequence_parallel_world_size())
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=get_sequence_parallel_group())
            
            optimizer.step()
            
            iter_t_list.append(time.time() - t0)
            loss_list.append(loss.item())
            
            lr_scheduler.step()
        
        # 每处理完一个图打印一次
        if args.rank == 0:
            print(f"Epoch {epoch} | Graph {graph_idx+1}/{len(packed_data)} | Loss: {loss.item():.4f}")
    
    return np.mean(loss_list)


@torch.no_grad()
def eval_gpu(args, model, device, packed_data, struct_info_list, criterion, evaluator, metric, str_prefix):
    model.eval()
    model.to(device)
    
    y_true_list, y_pred_list = [], []
    
    for graph_idx, (x, y, in_degree, out_degree, edge_index, num_nodes) in enumerate(packed_data):
        struct_info = struct_info_list[graph_idx]
        wm = struct_info.wm
        
        num_partitions = len(wm.partitioned_results)
        kv_cache_per_partition = [None] * num_partitions
        
        for part_idx in range(num_partitions):
            partition_nodes = wm.partitioned_results[part_idx]
            sub_edge_index = wm.sub_edge_index_for_partition_results[part_idx]
            
            x_i = x[partition_nodes, :].to(device)
            in_degree_i = struct_info.in_degree[partition_nodes].to(device)
            out_degree_i = struct_info.out_degree[partition_nodes].to(device)
            edge_index_i = sub_edge_index.to(device)
            
            current_kv_cache = kv_cache_per_partition[part_idx]
            
            pred, updated_kv_cache = model(
                x_i.unsqueeze(0),
                in_degree_i.unsqueeze(0),
                out_degree_i.unsqueeze(0),
                edge_index_i,
                attn_type=args.attn_type,
                dup_nodes_kv_cache=current_kv_cache,
                part_id=part_idx
            )
            
            if updated_kv_cache is not None:
                kv_cache_per_partition[part_idx] = updated_kv_cache
        
        y_true = y.to(device)
        y_pred = pred.argmax(1)
        
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
    
    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)
    
    eval_metric = calc_acc(y_true, y_pred)

    if args.rank == 0:
        print(f'{str_prefix} {metric}: {eval_metric}')
    
    return eval_metric


def main(rank, args, packed_train_data, packed_val_data, packed_test_data, n_classes, train_struct_info, val_struct_info, test_struct_info):
    # Note: Distributed initialization is already done in main_entry via initialize_distributed(args)
    # We just need to set the device based on local rank
    # pass

    # initialize_sequence_parallel(args.seq_len, 1, 1, args.sequence_parallel_size)
    
    device = f'cuda:{torch.cuda.current_device()}'
    
    # 模型选择 - 支持 GT 和 Graphormer
    if args.model == "graphormer":
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.input_dropout_rate,
            ffn_dim=args.ffn_dim,
            dataset_name=args.dataset,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            attention_dropout_rate=args.attention_dropout_rate,     
            output_dim=n_classes,
            args=args,
        ).to(device)
    elif args.model == "gt":
        model = GT(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.input_dropout_rate,
            ffn_dim=args.ffn_dim,
            dataset_name=args.dataset,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            attention_dropout_rate=args.attention_dropout_rate,     
            output_dim=n_classes,
            args=args,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}. Use 'graphormer' or 'gt'")
    
    if args.rank == 0:
        print(f'Model: {args.model}')
        print('Model params:', sum(p.numel() for p in model.parameters()))
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup=args.warmup_updates,
        total=args.total_updates,
        lr=args.lr,
        end_lr=args.end_lr,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            args, model, device, packed_train_data, train_struct_info,
            optimizer, criterion, epoch, lr_scheduler
        )
        
        val_acc = eval_gpu(
            args, model, device, packed_val_data, val_struct_info,
            criterion, None, "accuracy", "Val"
        )
        
        test_acc = eval_gpu(
            args, model, device, packed_test_data, test_struct_info,
            criterion, None, "accuracy", "Test"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        if args.rank == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Test Acc = {test_acc:.4f}")
            print(f"Best: Val = {best_val_acc:.4f}, Test = {best_test_acc:.4f}")
    
    dist.destroy_process_group()


def main_entry():
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    args = parser.parse_args()
    
    print(args)
    
    # 初始化分布式
    initialize_distributed(args)
    
    # 数据加载
    dm = GraphormerDataset(dataset_name=args.dataset, dataset_dir=args.dataset_dir,
                         num_workers=args.num_workers, batch_size=args.batch_size, seed=args.seed, 
                         multi_hop_max_dist=args.multi_hop_max_dist, spatial_pos_max=args.spatial_pos_max, myargs=args)
    dm.setup()
    n_classes = dm.dataset["num_class"]
    
    print(f"num_class: {n_classes}")
    
    train_loader, val_loader, test_loader = dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()
    
    # 打包数据并构建结构信息
    packed_train_data, packed_val_data, packed_test_data = [], [], []
    train_struct_info, val_struct_info, test_struct_info = [], [], []
    
    t0 = time.time()
    
    if args.rank == 0:
        print("Processing training data...")
    
    for batch_idx, batch in enumerate(train_loader):
        # 将 batch 拆分成多个独立的图
        graphs = split_batch(batch)
        
        # 对每个图单独构建结构信息
        for graph_idx, (x, y, in_degree, out_degree, edge_index, num_nodes) in enumerate(graphs):
            packed_train_data.append((
                x, y, in_degree, out_degree, edge_index, num_nodes
            ))
            
            struct_info = build_graph_struct_info(args, x, edge_index, num_nodes)
            train_struct_info.append(struct_info)
        
        if args.rank == 0 and (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(train_loader)} training batches, total graphs: {len(packed_train_data)}")
    
    if args.rank == 0:
        print(f"Train data prepared: {len(packed_train_data)} graphs, time: {time.time()-t0:.1f}s")
    
    # 处理验证集和测试集
    for batch in val_loader:
        graphs = split_batch(batch)
        for x, y, in_degree, out_degree, edge_index, num_nodes in graphs:
            packed_val_data.append((
                x, y, in_degree, out_degree, edge_index, num_nodes
            ))
            val_struct_info.append(build_graph_struct_info(args, x, edge_index, num_nodes))
    
    for batch in test_loader:
        graphs = split_batch(batch)
        for x, y, in_degree, out_degree, edge_index, num_nodes in graphs:
            packed_test_data.append((
                x, y, in_degree, out_degree, edge_index, num_nodes
            ))
            test_struct_info.append(build_graph_struct_info(args, x, edge_index, num_nodes))
    
    if args.rank == 0:
        print(f"Total data prepared time: {time.time()-t0:.1f}s")
        print(f"Train: {len(packed_train_data)}, Val: {len(packed_val_data)}, Test: {len(packed_test_data)}")
    
    # 直接在当前进程中执行训练（每个torchrun启动的进程处理自己的工作）
    main(args.rank, args, packed_train_data, packed_val_data, packed_test_data, n_classes, 
         train_struct_info, val_struct_info, test_struct_info)


if __name__ == "__main__":
    main_entry()
