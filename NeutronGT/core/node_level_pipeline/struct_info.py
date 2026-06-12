import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from core.metisPartition import weightMetis_keepParent
from core.ppr_preprocess import add_isolated_connections, build_adj_fromat, personal_pagerank
from gt_sp.utils import get_node_degrees

from .graph_data import _ensure_edge_index, _get_node_degrees_from_csr
from .preprocess_cache import compute_preprocess_cache_key, load_preprocess_cache, save_preprocess_cache
from .runtime import sync_device


class StructInfo:
    def __init__(self, **kwargs) -> None:
        self.graph_in_degree = kwargs["graph_in_degree"]
        self.graph_out_degree = kwargs["graph_out_degree"]
        self.sorted_ppr_matrix = kwargs["sorted_ppr_matrix"]
        self.wm = kwargs["wm"]
        self.graph_edge_index = kwargs.get("graph_edge_index")
        self.graph_csr_data = kwargs.get("graph_csr_data")
        self.num_nodes = kwargs.get("num_nodes")
        self.spatial_pos_by_pid = None
        self.sub_edge_index_list = None
        self.local_partition_ids = []
        self.local_partitions = []
        self.local_sub_edge_index_for_partition_results = []
        self.local_dup_nodes_per_partition = []
        self.local_spatial_pos_by_pid = []
        self.local_dup_indices = []
        self.local_dup_nodes_per_partition_feature = None
        self.global_partitioned_results_cpu = None
        self.global_sub_edge_index_for_partition_results_cpu = None
        self.window_state_version = 0


def build_placeholder_struct_info(graph_in_degree, graph_out_degree, edge_index=None, edge_csr_data=None, num_nodes=None):
    placeholder_wm = SimpleNamespace(
        partitioned_results=[],
        sub_edge_index_for_partition_results=[],
        dup_nodes_per_partition=[],
    )
    return StructInfo(
        graph_in_degree=graph_in_degree,
        graph_out_degree=graph_out_degree,
        sorted_ppr_matrix=None,
        wm=placeholder_wm,
        graph_edge_index=edge_index,
        graph_csr_data=edge_csr_data,
        num_nodes=num_nodes,
    )


def get_rank_source_range(num_nodes: int, rank: int, world_size: int):
    start = (num_nodes * rank) // world_size
    end = (num_nodes * (rank + 1)) // world_size
    return start, end


def gather_ppr_shards(local_ppr: tuple[torch.Tensor, torch.Tensor], rank: int, world_size: int):
    edge_index, edge_value = local_ppr
    if world_size <= 1:
        return edge_index, edge_value

    device = edge_index.device
    local_edge_count = torch.tensor([int(edge_value.numel())], dtype=torch.long, device=device)
    gathered_edge_counts = [torch.zeros_like(local_edge_count) for _ in range(world_size)]
    dist.all_gather(gathered_edge_counts, local_edge_count)
    edge_counts = [int(item.item()) for item in gathered_edge_counts]
    try:
        dist.barrier()
    except Exception as exc:
        raise RuntimeError(
            "Distributed PPR shard synchronization failed before gather. This usually means at least one rank did not finish local APPNP propagation cleanly."
        ) from exc

    if rank != 0:
        try:
            if edge_counts[rank] > 0:
                dist.send(edge_index.contiguous(), dst=0)
                dist.send(edge_value.contiguous(), dst=0)
        except Exception as exc:
            raise RuntimeError(
                "Distributed PPR shard send failed. Check whether rank 0 is still alive and earlier logs for sender-side failures."
            ) from exc
        return None

    edge_index_parts = []
    edge_value_parts = []
    for src_rank, shard_edge_count in enumerate(edge_counts):
        if shard_edge_count <= 0:
            continue
        if src_rank == 0:
            edge_index_parts.append(edge_index.detach().cpu())
            edge_value_parts.append(edge_value.detach().cpu())
            continue
        recv_edge_index = torch.empty((2, shard_edge_count), dtype=edge_index.dtype, device=device)
        recv_edge_value = torch.empty((shard_edge_count,), dtype=edge_value.dtype, device=device)
        try:
            dist.recv(recv_edge_index, src=src_rank)
            dist.recv(recv_edge_value, src=src_rank)
        except Exception as exc:
            raise RuntimeError(
                f"Distributed PPR shard receive failed while reading rank {src_rank}. Check sender-side logs for earlier failures."
            ) from exc
        edge_index_parts.append(recv_edge_index.cpu())
        edge_value_parts.append(recv_edge_value.cpu())
        del recv_edge_index, recv_edge_value

    if not edge_index_parts:
        empty_index = torch.empty((2, 0), dtype=torch.long)
        empty_value = torch.empty((0,), dtype=edge_value.dtype)
        return empty_index, empty_value
    return torch.cat(edge_index_parts, dim=1), torch.cat(edge_value_parts, dim=0)


def _preprocess_cache_enabled(args):
    return int(getattr(args, 'use_preprocess_cache', 1)) == 1 and int(getattr(args, 'use_cache', 0)) == 1


def _log_window_stats(partitioned_results):
    if not partitioned_results:
        print("window stats: no windows constructed", flush=True)
        return

    window_sizes = [int(part.numel()) for part in partitioned_results]
    total_window_nodes = int(sum(window_sizes))
    all_nodes = torch.cat([part.to(torch.long).cpu() for part in partitioned_results], dim=0)
    unique_window_nodes = int(torch.unique(all_nodes).numel()) if all_nodes.numel() > 0 else 0
    overlap_ratio = 1.0 - (unique_window_nodes / total_window_nodes) if total_window_nodes > 0 else 0.0

    print(
        "window stats: "
        f"count={len(window_sizes)}, "
        f"min={min(window_sizes)}, "
        f"max={max(window_sizes)}, "
        f"mean={sum(window_sizes) / len(window_sizes):.2f}, "
        f"total={total_window_nodes}, "
        f"unique={unique_window_nodes}, "
        f"overlap_ratio={overlap_ratio:.4f}",
        flush=True,
    )


def _build_struct_info_from_cache_payload(payload, graph_in_degree, graph_out_degree, edge_index=None, edge_csr_data=None, num_nodes=None):
    wm_data = payload['wm']
    wm = SimpleNamespace(
        partitioned_results=wm_data['partitioned_results'],
        sub_edge_index_for_partition_results=wm_data['sub_edge_index_for_partition_results'],
        dup_nodes_per_partition=wm_data['dup_nodes_per_partition'],
    )
    return StructInfo(
        graph_in_degree=payload.get('graph_in_degree', graph_in_degree),
        graph_out_degree=payload.get('graph_out_degree', graph_out_degree),
        sorted_ppr_matrix=payload.get('sorted_ppr_matrix'),
        wm=wm,
        graph_edge_index=payload.get('graph_edge_index', edge_index),
        graph_csr_data=payload.get('graph_csr_data', edge_csr_data),
        num_nodes=payload.get('num_nodes', num_nodes),
    )


def build_graph_struct_info(args, N, edge_index, feature, world_size, device, topk=50, n_parts=50,
                            related_nodes_topk_rate=5, connect_prob=0.01, edge_csr_data=None):
    # ------------模型位置编码所需的数据------------
    graph_in_degree, graph_out_degree = None, None
    if args.struct_enc == "True":
        if edge_csr_data is not None:
            graph_in_degree, graph_out_degree = _get_node_degrees_from_csr(edge_csr_data, N)
        else:
            graph_in_degree, graph_out_degree = get_node_degrees(edge_index, N)
    # ------------- preprocess cache ---------------
    cache_enabled = _preprocess_cache_enabled(args)
    if int(getattr(args, 'use_preprocess_cache', 1)) == 1 and int(getattr(args, 'use_cache', 0)) != 1 and args.rank == 0:
        print('Preprocess cache disabled: only supported for fixed-window training with --use_cache 1.')

    cache_key = None
    cache_path = None
    cache_args_snapshot = None
    distributed_appnp_ppr = world_size > 1 and args.ppr_backend == "appnp"

    if cache_enabled and int(getattr(args, 'refresh_preprocess_cache', 0)) != 1:
        if world_size > 1:
            if args.rank == 0:
                payload, cache_key, cache_path, cache_args_snapshot, cache_load_time, cache_size_mb = load_preprocess_cache(
                    args,
                    graph_in_degree,
                    graph_out_degree,
                    edge_index=edge_index,
                    edge_csr_data=edge_csr_data,
                    num_nodes=N,
                    world_size=world_size,
                )
                cache_hit = payload is not None
                if cache_hit:
                    print(f"Preprocess cache hit: path={cache_path}, key={cache_key[:12]}, load_time={cache_load_time:.3f}s, size_mb={cache_size_mb:.2f}")
                else:
                    print(f"Preprocess cache miss: path={cache_path}, key={cache_key[:12]}, load_time={cache_load_time:.3f}s, size_mb={cache_size_mb:.2f}")
                hit_box = [cache_hit]
                dist.broadcast_object_list(hit_box, src=0) # 广播是否 hit cache
            else:
                # 非 root GPU(0号) 生成空窗口信息，等待窗口状态同步
                hit_box = [False]
                dist.broadcast_object_list(hit_box, src=0)  
                if hit_box[0]:
                    return build_placeholder_struct_info(graph_in_degree, graph_out_degree, edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N)
            if args.rank == 0 and cache_hit:
                return _build_struct_info_from_cache_payload(payload, graph_in_degree, graph_out_degree, edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N)
        else:
            payload, cache_key, cache_path, cache_args_snapshot, cache_load_time, cache_size_mb = load_preprocess_cache(
                args,
                graph_in_degree,
                graph_out_degree,
                edge_index=edge_index,
                edge_csr_data=edge_csr_data,
                num_nodes=N,
                world_size=world_size,
            )
            if payload is not None:
                print(f"Preprocess cache hit: path={cache_path}, key={cache_key[:12]}, load_time={cache_load_time:.3f}s, size_mb={cache_size_mb:.2f}")
                return _build_struct_info_from_cache_payload(payload, graph_in_degree, graph_out_degree, edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N)
            print(f"Preprocess cache miss: path={cache_path}, key={cache_key[:12]}, load_time={cache_load_time:.3f}s, size_mb={cache_size_mb:.2f}")
    elif cache_enabled and args.rank == 0 and int(getattr(args, 'refresh_preprocess_cache', 0)) == 1:
        cache_key, cache_args_snapshot = compute_preprocess_cache_key(args, world_size)
        print(f"Preprocess cache miss: refresh requested, key={cache_key[:12]}")
    # 如果不是 APPNP 多卡预处理，那么只 rank0 去做窗口构建，其他 rank 生成占位符等待
    if world_size > 1 and not distributed_appnp_ppr and args.rank != 0:
        return build_placeholder_struct_info(graph_in_degree, graph_out_degree, edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N)
    # 如果是，那么为每个 rank 划分需要计算的 ppr 的源点
    source_start, source_end = 0, N
    if distributed_appnp_ppr:
        source_start, source_end = get_rank_source_range(N, args.rank, world_size)



        
    # ----------------------计算以每个点为源点的PPR --------------------------
    # 加权稀疏交互图    
    # sorted_ppr_matrix = (edge_index, edge_value): ([[src,src...src],[dest,dest,...dest]],[val,val...val])
    sync_device(device)
    local_ppr_start = time.time()
    print(
        f"rank {args.rank}: start local PPR, source_range=({source_start}, {source_end}), "
        f"backend={args.ppr_backend}, topk={topk}",
        flush=True,
    )
    local_sorted_ppr_matrix = personal_pagerank(
        edge_index,
        args.ppr_alpha,
        topk=topk,
        backend=args.ppr_backend,
        num_iterations=args.ppr_num_iterations,
        batch_size=args.ppr_batch_size,
        eps=args.ppr_eps,
        device=device,
        csr_data=edge_csr_data,
        num_nodes=N,
        iter_topk=args.ppr_iter_topk,
        source_start=source_start if distributed_appnp_ppr else None,
        source_end=source_end if distributed_appnp_ppr else None,
    )
    sync_device(device)
    local_ppr_time = time.time() - local_ppr_start
    local_edge_count = int(local_sorted_ppr_matrix[1].numel())
    print(
        f"rank {args.rank}: finish local PPR, local_edge_count={local_edge_count}, "
        f"time={local_ppr_time:.3f}s",
        flush=True,
    )

    sorted_ppr_matrix = local_sorted_ppr_matrix
    ppr_time = local_ppr_time
    gather_time = 0.0
    if distributed_appnp_ppr:
        sync_device(device)
        gather_start = time.time()
        print(f"rank {args.rank}: enter PPR gather", flush=True)
        # 把多个GPU并行计算的 PPR 数据 gather 在 rank 0 CPU (concat方式聚合为一张大图)
        sorted_ppr_matrix = gather_ppr_shards(local_sorted_ppr_matrix, rank=args.rank, world_size=world_size)
        sync_device(device)
        gather_time = time.time() - gather_start
        print(f"rank {args.rank}: finish PPR gather, time={gather_time:.3f}s", flush=True)
        if args.rank == 0 and sorted_ppr_matrix is not None:
            ppr_time += gather_time
        if args.rank != 0:
            return build_placeholder_struct_info(graph_in_degree, graph_out_degree, edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N)

    graph_edge_index = _ensure_edge_index(edge_index, edge_csr_data)

    # ---------------------------为 PPR 图中的孤立点补充随机连边 -----------------------------
    isolated_start = time.time()
    sorted_ppr_matrix = add_isolated_connections(sorted_ppr_matrix, graph_edge_index, N, connect_prob=connect_prob)
    isolated_time = time.time() - isolated_start
    # -------------------------- 将 PPR 图转换为划分所需要的 CSR 格式 -------------------------
    adj_build_start = time.time()
    csr_adjacency, eweights, _ = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)
    adj_build_time = time.time() - adj_build_start
    # -------------------------- 开始生成窗口 ----------------------------
    partition_build_start = time.time()
    if args.rank == 0:
        print(
            f"rank {args.rank}: start window construction, ppr_edges={int(sorted_ppr_matrix[1].numel())}",
            flush=True,
        )
    wm = weightMetis_keepParent(
        csr_adjacency=csr_adjacency,
        eweights=eweights,
        n_parts=n_parts,
        feature=feature,
        edge_index=graph_edge_index,
        related_nodes_topk_rate=related_nodes_topk_rate,
        attn_type=args.attn_type,
        sorted_ppr_matrix=sorted_ppr_matrix,
        random_replace_window_nodes=getattr(args, 'random_replace_window_nodes', 0),
        disable_window_node_expansion=getattr(args, 'disable_window_node_expansion', 0),
        high_degree_replace_window_nodes=getattr(args, 'high_degree_replace_window_nodes', 0),
    )
    partition_build_time = time.time() - partition_build_start
    if args.rank == 0:
        print(
            f"rank {args.rank}: finish window construction, total_windows={len(wm.partitioned_results)}, "
            f"time={partition_build_time:.3f}s",
            flush=True,
        )
        _log_window_stats(wm.partitioned_results)

    # Stage 1: PPR 到基础 Metis 划分完成。
    stage1_time = (
        ppr_time
        + isolated_time
        + adj_build_time
        + wm.timing_stats.get('parent_partition_time', 0.0)
        + wm.timing_stats.get('child_partition_time', 0.0)
    )
    # Stage 2: 基础 Metis 之后到 build_graph_struct_info 返回
    stage2_time = (
        wm.timing_stats.get('centroid_build_time', 0.0)
        + wm.timing_stats.get('related_nodes_merge_time', 0.0)
        + wm.timing_stats.get('feature_sim_merge_time', 0.0)
        + wm.timing_stats.get('expanded_edge_concat_time', 0.0)
        + wm.timing_stats.get('duplicate_rerange_time', 0.0)
        + wm.timing_stats.get('subgraph_build_time', 0.0)
    )
    print(f"Preprocess Stage 1 Time: {stage1_time:.3f}s")
    print(f"Preprocess Stage 2 Time: {stage2_time:.3f}s")

    # 此时 划分好的核心窗口集合在 rank 0 上
    struct_info = StructInfo(
        graph_in_degree=graph_in_degree,
        graph_out_degree=graph_out_degree,
        sorted_ppr_matrix=sorted_ppr_matrix,
        wm=wm,
        graph_edge_index=graph_edge_index,
        graph_csr_data=edge_csr_data,
        num_nodes=N,
    )

    if cache_enabled and args.rank == 0:
        if cache_key is None or cache_args_snapshot is None:
            cache_key, cache_args_snapshot = compute_preprocess_cache_key(args, world_size)
        saved_cache_path, cache_save_time, cache_size_mb = save_preprocess_cache(args, struct_info, cache_key, cache_args_snapshot)
        print(f"Preprocess cache save: path={saved_cache_path}, key={cache_key[:12]}, save_time={cache_save_time:.3f}s, size_mb={cache_size_mb:.2f}")

    return struct_info
