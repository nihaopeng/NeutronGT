import os
import time
from collections import deque

import torch
import torch.distributed as dist
from torch_geometric.utils import subgraph

from .struct_info import StructInfo


def _empty_cache(feature: torch.Tensor, device: str):
    return feature.new_empty((0, feature.shape[1]), device=device)


def _window_state_dir(args):
    cache_dir = os.path.join(args.dataset_dir, args.dataset, 'window_state_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _bundle_path(args, version: int, rank: int):
    return os.path.join(_window_state_dir(args), f'window_state_v{version}_rank{rank}.pt')


def _stash_global_window_state_cpu(structInfo: StructInfo):
    if structInfo.wm.partitioned_results:
        structInfo.global_partitioned_results_cpu = structInfo.wm.partitioned_results
    if structInfo.wm.sub_edge_index_for_partition_results:
        structInfo.global_sub_edge_index_for_partition_results_cpu = structInfo.wm.sub_edge_index_for_partition_results


def restore_global_window_state(structInfo: StructInfo):
    if structInfo.global_partitioned_results_cpu is not None:
        structInfo.wm.partitioned_results = structInfo.global_partitioned_results_cpu
    if structInfo.global_sub_edge_index_for_partition_results_cpu is not None:
        structInfo.wm.sub_edge_index_for_partition_results = structInfo.global_sub_edge_index_for_partition_results_cpu


def _release_hot_global_window_state(structInfo: StructInfo):
    structInfo.wm.partitioned_results = []
    structInfo.wm.sub_edge_index_for_partition_results = []
    structInfo.wm.dup_nodes_per_partition = []
    structInfo.spatial_pos_by_pid = []


def _assign_local_window_bundle(structInfo: StructInfo, bundle):
    structInfo.local_partition_ids = bundle['local_partition_ids']
    structInfo.local_partitions = bundle['local_partitions']


def _compute_local_duplicate_nodes(local_partitions):
    if not local_partitions:
        return [], []
    all_nodes = torch.cat([part.to(torch.long).cpu() for part in local_partitions]) if local_partitions else torch.empty((0,), dtype=torch.long)
    if all_nodes.numel() == 0:
        return [part.clone() for part in local_partitions], [torch.empty((0,), dtype=torch.long) for _ in local_partitions]

    unique_nodes, counts = torch.unique(all_nodes, return_counts=True, sorted=True)
    duplicated_nodes = unique_nodes[counts >= 2]
    if duplicated_nodes.numel() == 0:
        return [part.clone() for part in local_partitions], [torch.empty((0,), dtype=torch.long) for _ in local_partitions]

    reranged_partitions = []
    dup_nodes_per_partition = []
    for part in local_partitions:
        part_cpu = part.to(torch.long).cpu()
        dup_mask = torch.isin(part_cpu, duplicated_nodes)
        dup_nodes = part_cpu[dup_mask]
        non_dup_nodes = part_cpu[~dup_mask]
        reranged_partitions.append(torch.cat([dup_nodes, non_dup_nodes], dim=0))
        dup_nodes_per_partition.append(dup_nodes)
    return reranged_partitions, dup_nodes_per_partition


def _build_local_ppr_sub_edge_index_list(structInfo: StructInfo, local_partitions):
    if structInfo.sorted_ppr_matrix is None:
        return []
    ppr_edge_index = structInfo.sorted_ppr_matrix[0].to('cpu')
    local_ppr_sub_edge_index_list = []
    for partition in local_partitions:
        local_edge_index, _ = subgraph(
            partition,
            ppr_edge_index,
            relabel_nodes=True,
            num_nodes=structInfo.num_nodes,
        )
        local_ppr_sub_edge_index_list.append(local_edge_index)
    return local_ppr_sub_edge_index_list


def _subgraph_from_csr(node_set: torch.Tensor, rowptr: torch.Tensor, col: torch.Tensor):
    node_set = node_set.to(torch.long).cpu()
    if node_set.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    sort_order = torch.argsort(node_set)
    sorted_nodes = node_set[sort_order]
    row_chunks = []
    col_chunks = []
    for local_src, global_src in enumerate(node_set.tolist()):
        start = int(rowptr[global_src].item())
        end = int(rowptr[global_src + 1].item())
        neighbors = col[start:end].to(torch.long)
        if neighbors.numel() == 0:
            continue
        positions = torch.searchsorted(sorted_nodes, neighbors)
        in_bounds = positions < sorted_nodes.numel()
        matched = torch.zeros_like(in_bounds, dtype=torch.bool)
        if in_bounds.any():
            matched[in_bounds] = sorted_nodes[positions[in_bounds]] == neighbors[in_bounds]
        if not matched.any():
            continue
        local_dst = sort_order[positions[matched]].to(torch.long)
        row_chunks.append(torch.full((local_dst.numel(),), local_src, dtype=torch.long))
        col_chunks.append(local_dst)
    if not row_chunks:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack([torch.cat(row_chunks), torch.cat(col_chunks)], dim=0)


def _build_local_sub_edge_index_list(structInfo: StructInfo, local_partitions):
    local_sub_edge_index_list = []
    if structInfo.graph_edge_index is not None:
        edge_index = structInfo.graph_edge_index.to('cpu')
        for partition in local_partitions:
            local_edge_index, _ = subgraph(
                partition,
                edge_index,
                relabel_nodes=True,
                num_nodes=structInfo.num_nodes,
            )
            local_sub_edge_index_list.append(local_edge_index)
        return local_sub_edge_index_list

    rowptr = structInfo.graph_csr_data['rowptr'].to(torch.long).cpu()
    col = structInfo.graph_csr_data['col'].to(torch.long).cpu()
    for partition in local_partitions:
        local_sub_edge_index_list.append(_subgraph_from_csr(partition, rowptr, col))
    return local_sub_edge_index_list


def _compute_local_spatial_pos(local_partitions, local_ppr_sub_edge_index_list, max_dist: int):
    spatial_pos_list = []
    for partition, local_edge_index in zip(local_partitions, local_ppr_sub_edge_index_list):
        num_nodes = int(partition.numel())
        dist_mat = torch.full((num_nodes, num_nodes), max_dist + 1, dtype=torch.long)
        if num_nodes == 0:
            spatial_pos_list.append(dist_mat)
            continue
        for i in range(num_nodes):
            dist_mat[i, i] = 0
        adjacency = [[] for _ in range(num_nodes)]
        if local_edge_index.numel() > 0:
            src = local_edge_index[0].tolist()
            dst = local_edge_index[1].tolist()
            for u, v in zip(src, dst):
                adjacency[u].append(v)
        for src in range(num_nodes):
            queue = deque([src])
            while queue:
                u = queue.popleft()
                current_dist = int(dist_mat[src, u].item())
                if current_dist >= max_dist:
                    continue
                for v in adjacency[u]:
                    if dist_mat[src, v] > current_dist + 1:
                        dist_mat[src, v] = current_dist + 1
                        queue.append(v)
        spatial_pos = torch.zeros_like(dist_mat)
        reachable = dist_mat <= max_dist
        spatial_pos[reachable] = dist_mat[reachable] + 1
        spatial_pos_list.append(spatial_pos)
    return spatial_pos_list


def build_local_partitions(structInfo: StructInfo, rank: int, world_size: int):
    return structInfo.local_partition_ids, structInfo.local_partitions


def build_dup_cache_metadata(structInfo: StructInfo, feature: torch.Tensor, device: str):
    local_dup_nodes = getattr(structInfo, 'local_dup_nodes_per_partition', None) or []
    if not local_dup_nodes:
        structInfo.local_dup_indices = []
        structInfo.local_dup_nodes_per_partition_feature = _empty_cache(feature, device)
        return torch.empty((0,), dtype=torch.long)

    non_empty_dup_nodes = [dup.to(torch.long).cpu() for dup in local_dup_nodes if dup.numel() > 0]
    if not non_empty_dup_nodes:
        structInfo.local_dup_indices = [torch.empty((0,), dtype=torch.long, device=device) for _ in local_dup_nodes]
        structInfo.local_dup_nodes_per_partition_feature = _empty_cache(feature, device)
        return torch.empty((0,), dtype=torch.long)

    dup_unique_sorted = torch.unique(torch.cat(non_empty_dup_nodes), sorted=True)
    structInfo.local_dup_indices = []
    for dup_nodes in local_dup_nodes:
        dup_nodes_cpu = dup_nodes.to(torch.long).cpu()
        if dup_nodes_cpu.numel() == 0:
            indices = torch.empty((0,), dtype=torch.long, device=device)
        else:
            indices = torch.searchsorted(dup_unique_sorted, dup_nodes_cpu).to(device=device, dtype=torch.long)
        structInfo.local_dup_indices.append(indices)

    structInfo.local_dup_nodes_per_partition_feature = feature[dup_unique_sorted].to(device)
    return dup_unique_sorted


def _build_local_bundle_for_rank(args, structInfo: StructInfo, rank: int):
    wm = structInfo.wm
    local_partition_ids = list(range(rank, len(wm.partitioned_results), args.world_size))
    local_partitions = [wm.partitioned_results[pid].to(torch.long).cpu() for pid in local_partition_ids]
    if args.use_cache:
        local_partitions, _ = _compute_local_duplicate_nodes(local_partitions)
    bundle = {
        'local_partition_ids': local_partition_ids,
        'local_partitions': local_partitions,
    }
    if args.struct_enc == 'True':
        bundle['local_ppr_sub_edge_index_list'] = _build_local_ppr_sub_edge_index_list(structInfo, local_partitions)
    return bundle


def _rebuild_local_window_structures(args, structInfo: StructInfo, feature: torch.Tensor, device: str, local_ppr_sub_edge_index_list):
    """
    根据rank所在的分区构建cache
    """
    timing_stats = {
        'local_dup_cache_rebuild_time': 0.0,
        'local_subgraph_rebuild_time': 0.0,
        'local_spatial_rebuild_time': 0.0,
    }

    dup_start = time.time()
    if args.use_cache:
        local_partitions, local_dup_nodes_per_partition = _compute_local_duplicate_nodes(structInfo.local_partitions)
        structInfo.local_partitions = local_partitions
        structInfo.local_dup_nodes_per_partition = local_dup_nodes_per_partition
        build_dup_cache_metadata(structInfo, feature, device)
    else:
        structInfo.local_dup_nodes_per_partition = [torch.empty((0,), dtype=torch.long) for _ in structInfo.local_partitions]
        structInfo.local_dup_indices = []
        structInfo.local_dup_nodes_per_partition_feature = _empty_cache(feature, device)
    timing_stats['local_dup_cache_rebuild_time'] = time.time() - dup_start

    subgraph_start = time.time()
    structInfo.local_sub_edge_index_for_partition_results = _build_local_sub_edge_index_list(structInfo, structInfo.local_partitions)
    timing_stats['local_subgraph_rebuild_time'] = time.time() - subgraph_start

    spatial_start = time.time()
    if args.struct_enc == 'True':
        structInfo.local_spatial_pos_by_pid = _compute_local_spatial_pos(
            structInfo.local_partitions,
            local_ppr_sub_edge_index_list,
            max_dist=args.max_dist,
        )
    else:
        structInfo.local_spatial_pos_by_pid = []
    timing_stats['local_spatial_rebuild_time'] = time.time() - spatial_start
    return timing_stats


def broadcast_window_state(args, structInfo: StructInfo, feature: torch.Tensor, device: str):
    timing_stats = {
        'bundle_write_time': 0.0,
        'bundle_load_time': 0.0,
        'local_dup_cache_rebuild_time': 0.0,
        'local_subgraph_rebuild_time': 0.0,
        'local_spatial_rebuild_time': 0.0,
        'window_state_total_time': 0.0,
    }
    overall_start = time.time()

    if args.rank == 0:
        # 把全局窗口结构保存在 CPU
        restore_global_window_state(structInfo)
        _stash_global_window_state_cpu(structInfo)

    if args.world_size <= 1:
        bundle = _build_local_bundle_for_rank(args, structInfo, args.rank)
        _assign_local_window_bundle(structInfo, bundle)
        local_ppr_sub_edge_index_list = bundle.get('local_ppr_sub_edge_index_list', [])
        rebuild_stats = _rebuild_local_window_structures(args, structInfo, feature, device, local_ppr_sub_edge_index_list)
        timing_stats.update(rebuild_stats)
        if args.rank == 0:
            _release_hot_global_window_state(structInfo)
        timing_stats['window_state_total_time'] = time.time() - overall_start
        return timing_stats

    version = structInfo.window_state_version

    # 对每个 rank 生成一个 bundle,including
    # local_partition_ids、local_partitions、如果开结构编码，再加 local_ppr_sub_edge_index_list
    # 写到磁盘
    if args.rank == 0:
        bundle_write_start = time.time()
        for rank in range(args.world_size):
            bundle = _build_local_bundle_for_rank(args, structInfo, rank)
            torch.save(bundle, _bundle_path(args, version, rank))
        timing_stats['bundle_write_time'] = time.time() - bundle_write_start
    dist.barrier()

    # 每个 rank 读自己的 bundle，并本地重建结构
    bundle_load_start = time.time()
    bundle = torch.load(_bundle_path(args, version, args.rank), map_location='cpu')
    timing_stats['bundle_load_time'] = time.time() - bundle_load_start
    _assign_local_window_bundle(structInfo, bundle)
    local_ppr_sub_edge_index_list = bundle.get('local_ppr_sub_edge_index_list', [])
    rebuild_stats = _rebuild_local_window_structures(args, structInfo, feature, device, local_ppr_sub_edge_index_list)
    timing_stats.update(rebuild_stats)

    dist.barrier()

    #  rank 0 删除这些临时 bundle 文件，并释放全局窗口对象
    if args.rank == 0:
        for rank in range(args.world_size):
            path = _bundle_path(args, version, rank)
            if os.path.exists(path):
                os.remove(path)
    if args.rank == 0:
        _release_hot_global_window_state(structInfo)
    timing_stats['window_state_total_time'] = time.time() - overall_start
    return timing_stats
