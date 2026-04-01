import torch
import torch.distributed as dist

from .struct_info import StructInfo


def _empty_cache(feature: torch.Tensor, device: str):
    return feature.new_empty((0, feature.shape[1]), device=device)


def _slice_local_window_bundle(structInfo: StructInfo, rank: int, world_size: int):
    wm = structInfo.wm
    local_partition_ids = list(range(rank, len(wm.partitioned_results), world_size))
    local_partitions = [wm.partitioned_results[pid] for pid in local_partition_ids]
    local_sub_edge_index_list = [wm.sub_edge_index_for_partition_results[pid] for pid in local_partition_ids]
    local_dup_nodes_per_partition = [wm.dup_nodes_per_partition[pid] for pid in local_partition_ids]
    local_spatial_pos_by_pid = [structInfo.spatial_pos_by_pid[pid] for pid in local_partition_ids]
    return {
        "local_partition_ids": local_partition_ids,
        "local_partitions": local_partitions,
        "local_sub_edge_index_for_partition_results": local_sub_edge_index_list,
        "local_dup_nodes_per_partition": local_dup_nodes_per_partition,
        "local_spatial_pos_by_pid": local_spatial_pos_by_pid,
    }


def _assign_local_window_bundle(structInfo: StructInfo, bundle):
    structInfo.local_partition_ids = bundle["local_partition_ids"]
    structInfo.local_partitions = bundle["local_partitions"]
    structInfo.local_sub_edge_index_for_partition_results = bundle["local_sub_edge_index_for_partition_results"]
    structInfo.local_dup_nodes_per_partition = bundle["local_dup_nodes_per_partition"]
    structInfo.local_spatial_pos_by_pid = bundle["local_spatial_pos_by_pid"]


def build_local_partitions(structInfo: StructInfo, rank: int, world_size: int):
    if structInfo.local_partition_ids and structInfo.local_partitions:
        return structInfo.local_partition_ids, structInfo.local_partitions
    bundle = _slice_local_window_bundle(structInfo, rank, world_size)
    _assign_local_window_bundle(structInfo, bundle)
    return structInfo.local_partition_ids, structInfo.local_partitions


def build_dup_cache_metadata(structInfo: StructInfo, feature: torch.Tensor, device: str):
    local_dup_nodes = getattr(structInfo, "local_dup_nodes_per_partition", None) or []
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


def broadcast_window_state(args, structInfo: StructInfo, feature: torch.Tensor, device: str):
    if args.world_size <= 1:
        bundle = _slice_local_window_bundle(structInfo, args.rank, 1)
        _assign_local_window_bundle(structInfo, bundle)
        if args.use_cache:
            build_dup_cache_metadata(structInfo, feature, device)
        return

    payload = [None]
    scatter_list = None
    if args.rank == 0:
        scatter_list = [
            _slice_local_window_bundle(structInfo, rank, args.world_size)
            for rank in range(args.world_size)
        ]
    dist.scatter_object_list(payload, scatter_list, src=0)
    bundle = payload[0]
    _assign_local_window_bundle(structInfo, bundle)

    if args.rank != 0:
        structInfo.wm.partitioned_results = []
        structInfo.wm.sub_edge_index_for_partition_results = []
        structInfo.wm.dup_nodes_per_partition = []
        structInfo.spatial_pos_by_pid = []

    if args.use_cache:
        build_dup_cache_metadata(structInfo, feature, device)
