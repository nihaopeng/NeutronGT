import time
import os
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


def _ppr_shard_dir(dataset_dir: str, dataset: str) -> str:
    d = os.path.join(dataset_dir, dataset, 'ppr_temp')
    os.makedirs(d, exist_ok=True)
    return d


def _ppr_shard_path(dataset_dir: str, dataset: str, rank: int, run_id: str = 'default') -> str:
    return os.path.join(_ppr_shard_dir(dataset_dir, dataset), f'ppr_shard_{run_id}_{rank}.pt')


def _wait_for_path(path: str, timeout_seconds: float, poll_seconds: float = 30.0):
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for file: {path}")
        time.sleep(poll_seconds)


def _atomic_torch_save(obj, path: str):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def gather_ppr_shards(local_ppr: tuple[torch.Tensor, torch.Tensor], rank: int,
                       world_size: int, dataset_dir: str = '', dataset: str = '',
                       run_id: str = 'default', timeout_seconds: float | None = None):
    edge_index, edge_value = local_ppr
    if world_size <= 1:
        return edge_index, edge_value

    # 各 rank 将 PPR 结果写入磁盘，避免 dist.gather_object 在 rank 0 上
    # 同时持有所有 shard 的内存拷贝（papers100M: 4 shard ≈ 22 GB）
    value_dtype = edge_value.dtype  # 在 del 前保存，空 shard 路径需要用到
    shard_path = _ppr_shard_path(dataset_dir, dataset, rank, run_id=run_id)
    _atomic_torch_save((edge_index.cpu(), edge_value.cpu()), shard_path)
    del edge_index, edge_value

    if rank != 0:
        return None

    timeout_seconds = (24 * 60 * 60.0) if timeout_seconds is None else float(timeout_seconds)
    for r in range(world_size):
        _wait_for_path(_ppr_shard_path(dataset_dir, dataset, r, run_id=run_id), timeout_seconds=timeout_seconds)

    # rank 0 逐个从磁盘加载 shard 并增量合并，每个 shard 加载后立即释放
    result_edge_index = None
    result_edge_value = None
    for r in range(world_size):
        shard_path_r = _ppr_shard_path(dataset_dir, dataset, r, run_id=run_id)
        try:
            shard_ei, shard_ev = torch.load(shard_path_r, map_location='cpu')
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load PPR shard from rank {r} at {shard_path_r}"
            ) from exc
        if shard_ev.numel() <= 0:
            del shard_ei, shard_ev
            continue
        if result_edge_index is None:
            result_edge_index = shard_ei
            result_edge_value = shard_ev
        else:
            result_edge_index = torch.cat([result_edge_index, shard_ei], dim=1)
            result_edge_value = torch.cat([result_edge_value, shard_ev], dim=0)
            del shard_ei, shard_ev

    # 清理临时文件
    for r in range(world_size):
        p = _ppr_shard_path(dataset_dir, dataset, r, run_id=run_id)
        if os.path.exists(p):
            os.remove(p)
    shard_dir = _ppr_shard_dir(dataset_dir, dataset)
    if os.path.isdir(shard_dir) and not os.listdir(shard_dir):
        os.rmdir(shard_dir)

    if result_edge_index is None:
        empty_index = torch.empty((2, 0), dtype=torch.long)
        empty_value = torch.empty((0,), dtype=value_dtype)
        return empty_index, empty_value
    return result_edge_index, result_edge_value


def _preprocess_cache_enabled(args):
    return int(getattr(args, 'use_preprocess_cache', 1)) == 1 and int(getattr(args, 'use_cache', 0)) == 1


def _print_cache_hit_preprocess_timing(total_wall_time: float, cache_lookup_load_time: float, refresh_preprocess_cache: int):
    print(f"[PreprocessTiming] total_wall={total_wall_time:.3f}")
    print(
        f"[PreprocessTiming] cache hit=True "
        f"lookup_load={cache_lookup_load_time:.3f} "
        f"save=0.000 refresh={refresh_preprocess_cache}"
    )
    print("[PreprocessTiming] cache_hit=True; stage timings skipped because preprocess cache was reused.")


def _print_preprocess_timing(args, timing: dict, wm_timing_stats: dict):
    ppr_compute_time = timing.get('ppr_compute_time', 0.0)
    ppr_gather_time = timing.get('ppr_gather_time', 0.0)
    isolated_time = timing.get('isolated_time', 0.0)
    adj_build_time = timing.get('adj_build_time', 0.0)
    parent_partition_time = wm_timing_stats.get('parent_partition_time', 0.0)
    child_partition_time = wm_timing_stats.get('child_partition_time', 0.0)

    stage1_base_time = ppr_compute_time + ppr_gather_time + isolated_time + adj_build_time
    stage1_metis_time = parent_partition_time + child_partition_time
    stage1_time = stage1_base_time + stage1_metis_time
    stage2_time = (
        wm_timing_stats.get('centroid_build_time', 0.0)
        + wm_timing_stats.get('related_nodes_merge_time', 0.0)
        + wm_timing_stats.get('feature_sim_merge_time', 0.0)
        + wm_timing_stats.get('hub_node_merge_time', 0.0)
        + wm_timing_stats.get('random_fill_time', 0.0)
        + wm_timing_stats.get('expanded_edge_concat_time', 0.0)
        + wm_timing_stats.get('duplicate_rerange_time', 0.0)
        + wm_timing_stats.get('subgraph_build_time', 0.0)
    )

    print(f"[PreprocessTiming] total_wall={timing.get('total_wall_time', 0.0):.3f}")
    print(
        f"[PreprocessTiming] cache hit={timing.get('cache_hit', False)} "
        f"lookup_load={timing.get('cache_lookup_load_time', 0.0):.3f} "
        f"save={timing.get('cache_save_time', 0.0):.3f} "
        f"refresh={int(getattr(args, 'refresh_preprocess_cache', 0))}"
    )
    print(
        f"[PreprocessTiming] stage1_base "
        f"ppr_compute={ppr_compute_time:.3f} "
        f"ppr_gather={ppr_gather_time:.3f} "
        f"isolated={isolated_time:.3f} "
        f"build_adj={adj_build_time:.3f} "
        f"total={stage1_base_time:.3f}"
    )
    print(
        f"[PreprocessTiming] stage1_metis "
        f"parent={parent_partition_time:.3f} "
        f"child={child_partition_time:.3f} "
        f"total={stage1_metis_time:.3f}"
    )
    print(
        f"[PreprocessTiming] stage2_window "
        f"centroid={wm_timing_stats.get('centroid_build_time', 0.0):.3f} "
        f"related={wm_timing_stats.get('related_nodes_merge_time', 0.0):.3f} "
        f"feature={wm_timing_stats.get('feature_sim_merge_time', 0.0):.3f} "
        f"hub={wm_timing_stats.get('hub_node_merge_time', 0.0):.3f} "
        f"filler={wm_timing_stats.get('random_fill_time', 0.0):.3f} "
        f"expanded_edge={wm_timing_stats.get('expanded_edge_concat_time', 0.0):.3f} "
        f"duplicate_rerange={wm_timing_stats.get('duplicate_rerange_time', 0.0):.3f} "
        f"subgraph={wm_timing_stats.get('subgraph_build_time', 0.0):.3f} "
        f"total={stage2_time:.3f}"
    )

    target_extra_nodes = int(wm_timing_stats.get('augmentation_target_extra_nodes', 0))
    related_nodes = int(wm_timing_stats.get('augmentation_related_nodes', 0))
    feature_nodes = int(wm_timing_stats.get('augmentation_feature_nodes', 0))
    hub_nodes = int(wm_timing_stats.get('augmentation_hub_nodes', 0))
    filler_nodes = int(wm_timing_stats.get('augmentation_filler_nodes', 0))
    filler_ratio = (filler_nodes / target_extra_nodes) if target_extra_nodes > 0 else 0.0
    print(
        f"[PreprocessTiming] window_aug={getattr(args, 'window_aug_strategy', 'ours')} "
        f"target_extra_nodes={target_extra_nodes} "
        f"related_nodes={related_nodes} "
        f"feature_nodes={feature_nodes} "
        f"hub_nodes={hub_nodes} "
        f"filler_nodes={filler_nodes} "
        f"filler_ratio={filler_ratio:.6f}"
    )
    print(f"Preprocess Stage 1 Time: {stage1_time:.3f}s")
    print(f"Preprocess Stage 2 Time: {stage2_time:.3f}s")


def _build_struct_info_from_cache_payload(payload, graph_in_degree, graph_out_degree, edge_index=None, edge_csr_data=None, num_nodes=None):
    wm_data = payload['wm']
    wm = SimpleNamespace(
        partitioned_results=wm_data['partitioned_results'],
        sub_edge_index_for_partition_results=wm_data['sub_edge_index_for_partition_results'],
        dup_nodes_per_partition=wm_data['dup_nodes_per_partition'],
    )
    cached_graph_edge_index = payload.get('graph_edge_index')
    cached_graph_csr_data = payload.get('graph_csr_data')
    return StructInfo(
        graph_in_degree=payload.get('graph_in_degree', graph_in_degree),
        graph_out_degree=payload.get('graph_out_degree', graph_out_degree),
        sorted_ppr_matrix=payload.get('sorted_ppr_matrix'),
        wm=wm,
        graph_edge_index=cached_graph_edge_index if cached_graph_edge_index is not None else edge_index,
        graph_csr_data=cached_graph_csr_data if cached_graph_csr_data is not None else edge_csr_data,
        num_nodes=payload.get('num_nodes', num_nodes),
    )


def build_graph_struct_info(args, N, edge_index, feature, world_size, device, topk=50, n_parts=50,
                            connect_prob=0.01, edge_csr_data=None):
    preprocess_start = time.time()
    graph_in_degree, graph_out_degree = None, None
    if args.struct_enc == "True":
        if edge_csr_data is not None:
            graph_in_degree, graph_out_degree = _get_node_degrees_from_csr(edge_csr_data, N)
        else:
            graph_in_degree, graph_out_degree = get_node_degrees(edge_index, N)

    cache_enabled = _preprocess_cache_enabled(args)
    if int(getattr(args, 'use_preprocess_cache', 1)) == 1 and int(getattr(args, 'use_cache', 0)) != 1 and args.rank == 0:
        print('Preprocess cache disabled: only supported for fixed-window training with --use_cache 1.')

    cache_key = None
    cache_path = None
    cache_args_snapshot = None
    cache_lookup_load_time = 0.0
    distributed_appnp_ppr = world_size > 1 and args.ppr_backend == "appnp"

    if cache_enabled and int(getattr(args, 'refresh_preprocess_cache', 0)) != 1:
        if world_size > 1:
            if args.rank == 0:
                payload, cache_key, cache_path, cache_args_snapshot, cache_load_time = load_preprocess_cache(
                    args,
                    graph_in_degree,
                    graph_out_degree,
                    edge_index=edge_index,
                    edge_csr_data=edge_csr_data,
                    num_nodes=N,
                    world_size=world_size,
                )
                cache_lookup_load_time = cache_load_time
                cache_hit = payload is not None
                if cache_hit:
                    print(f"Preprocess cache hit: path={cache_path}, key={cache_key[:12]}, load_time={cache_load_time:.3f}s")
                else:
                    print(f"Preprocess cache miss: path={cache_path}, key={cache_key[:12]}, load_time={cache_load_time:.3f}s")
                hit_box = [cache_hit]
                dist.broadcast_object_list(hit_box, src=0)
                if cache_hit:
                    _print_cache_hit_preprocess_timing(
                        total_wall_time=time.time() - preprocess_start,
                        cache_lookup_load_time=cache_lookup_load_time,
                        refresh_preprocess_cache=int(getattr(args, 'refresh_preprocess_cache', 0)),
                    )
                    return _build_struct_info_from_cache_payload(payload, graph_in_degree, graph_out_degree, edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N)
            else:
                hit_box = [False]
                dist.broadcast_object_list(hit_box, src=0)
                if hit_box[0]:
                    return build_placeholder_struct_info(graph_in_degree, graph_out_degree, edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N)
        else:
            payload, cache_key, cache_path, cache_args_snapshot, cache_load_time = load_preprocess_cache(
                args,
                graph_in_degree,
                graph_out_degree,
                edge_index=edge_index,
                edge_csr_data=edge_csr_data,
                num_nodes=N,
                world_size=world_size,
            )
            cache_lookup_load_time = cache_load_time
            if payload is not None:
                print(f"Preprocess cache hit: path={cache_path}, key={cache_key[:12]}, load_time={cache_load_time:.3f}s")
                _print_cache_hit_preprocess_timing(
                    total_wall_time=time.time() - preprocess_start,
                    cache_lookup_load_time=cache_lookup_load_time,
                    refresh_preprocess_cache=int(getattr(args, 'refresh_preprocess_cache', 0)),
                )
                return _build_struct_info_from_cache_payload(payload, graph_in_degree, graph_out_degree, edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N)
            print(f"Preprocess cache miss: path={cache_path}, key={cache_key[:12]}, load_time={cache_load_time:.3f}s")
    elif cache_enabled and args.rank == 0 and int(getattr(args, 'refresh_preprocess_cache', 0)) == 1:
        cache_key, cache_args_snapshot = compute_preprocess_cache_key(args, world_size)
        print(f"Preprocess cache miss: refresh requested, key={cache_key[:12]}")

    if world_size > 1 and not distributed_appnp_ppr and args.rank != 0:
        return build_placeholder_struct_info(graph_in_degree, graph_out_degree, edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N)

    source_start, source_end = 0, N
    if distributed_appnp_ppr:
        source_start, source_end = get_rank_source_range(N, args.rank, world_size)

    sync_device(device)
    local_ppr_start = time.time()
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
    ppr_compute_time = local_ppr_time
    local_edge_count = int(local_sorted_ppr_matrix[1].numel())

    sorted_ppr_matrix = local_sorted_ppr_matrix
    gather_time = 0.0
    if distributed_appnp_ppr:
        if args.rank == 0:
            print(f'[Preprocess] PPR computation done ({ppr_compute_time:.1f}s), gathering shards from disk...')
        sync_device(device)
        gather_start = time.time()
        wait_timeout = max(float(getattr(args, 'distributed_timeout_minutes', 10)) * 60.0 * 4.0, 24 * 60 * 60.0)
        sorted_ppr_matrix = gather_ppr_shards(local_sorted_ppr_matrix, rank=args.rank, world_size=world_size,
                                                 dataset_dir=args.dataset_dir, dataset=args.dataset,
                                                 run_id=getattr(args, 'sync_run_id', 'default'),
                                                 timeout_seconds=wait_timeout)
        sync_device(device)
        gather_time = time.time() - gather_start
        if args.rank != 0:
            return build_placeholder_struct_info(graph_in_degree, graph_out_degree, edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N)

    graph_edge_index = _ensure_edge_index(edge_index, edge_csr_data)

    if args.rank == 0:
        print(f'[Preprocess] PPR gather done, checking isolated connections...')

    isolated_start = time.time()
    sorted_ppr_matrix = add_isolated_connections(sorted_ppr_matrix, graph_edge_index, N, connect_prob=connect_prob)
    isolated_time = time.time() - isolated_start

    if args.rank == 0:
        print(f'[Preprocess] isolated check done ({isolated_time:.1f}s), building adjacency...')

    adj_build_start = time.time()
    csr_adjacency, eweights, _ = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)
    adj_build_time = time.time() - adj_build_start

    if args.rank == 0:
        print(f'[Preprocess] build_adj_fromat done ({adj_build_time:.1f}s), building Metis partitions...')

    partition_build_start = time.time()
    wm = weightMetis_keepParent(
        csr_adjacency=csr_adjacency,
        eweights=eweights,
        n_parts=n_parts,
        feature=feature,
        edge_index=graph_edge_index,
        edge_csr_data=edge_csr_data,
        attn_type=args.attn_type,
        sorted_ppr_matrix=sorted_ppr_matrix,
        window_aug_strategy=getattr(args, 'window_aug_strategy', 'ours'),
        window_extra_node_ratio=getattr(args, 'window_extra_node_ratio', 0.30),
        window_related_ratio=getattr(args, 'window_related_ratio', 0.12),
        window_feature_ratio=getattr(args, 'window_feature_ratio', 0.06),
        window_hub_ratio=getattr(args, 'window_hub_ratio', 0.12),
        feature_sim_virtual_edges_per_node=getattr(args, 'feature_sim_virtual_edges_per_node', 4),
        seed=getattr(args, 'seed', 42),
    )
    partition_build_time = time.time() - partition_build_start
    if args.rank == 0:
        print(f'[Preprocess] weightMetis_keepParent done ({partition_build_time:.1f}s)')
    # wm 已构造完毕，释放 build_adj_fromat 产物（PPR 矩阵后续 StructInfo 还需要）
    del csr_adjacency, eweights

    struct_info = StructInfo(
        graph_in_degree=graph_in_degree,
        graph_out_degree=graph_out_degree,
        sorted_ppr_matrix=sorted_ppr_matrix,
        wm=wm,
        graph_edge_index=graph_edge_index,
        graph_csr_data=edge_csr_data,
        num_nodes=N,
    )

    cache_save_time = 0.0
    if cache_enabled and args.rank == 0:
        if cache_key is None or cache_args_snapshot is None:
            cache_key, cache_args_snapshot = compute_preprocess_cache_key(args, world_size)
        saved_cache_path, cache_save_time = save_preprocess_cache(args, struct_info, cache_key, cache_args_snapshot)
        print(f"Preprocess cache save: path={saved_cache_path}, key={cache_key[:12]}, save_time={cache_save_time:.3f}s")

    preprocess_timing = {
        'cache_hit': False,
        'cache_lookup_load_time': cache_lookup_load_time,
        'cache_save_time': cache_save_time,
        'ppr_compute_time': ppr_compute_time,
        'ppr_gather_time': gather_time,
        'isolated_time': isolated_time,
        'adj_build_time': adj_build_time,
        'total_wall_time': time.time() - preprocess_start,
    }
    if args.rank == 0:
        _print_preprocess_timing(args, preprocess_timing, wm.timing_stats)

    # 释放 full PPR matrix：struct_enc=False 时不使用，节省 ~11 GB
    if args.struct_enc != "True":
        struct_info.sorted_ppr_matrix = None

    return struct_info
