import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from core.metisPartition import weightMetis_keepParent
from core.ppr_preprocess import add_isolated_connections, build_adj_fromat, personal_pagerank
from gt_sp.utils import get_node_degrees

from .graph_data import _ensure_edge_index, _get_node_degrees_from_csr
from .runtime import sync_device


class StructInfo:
    def __init__(self,**kwargs) -> None:
        self.graph_in_degree = kwargs["graph_in_degree"]
        self.graph_out_degree = kwargs["graph_out_degree"]
        self.sorted_ppr_matrix = kwargs["sorted_ppr_matrix"]  # tuple[torch.Tensor, torch.Tensor]
        self.wm = kwargs["wm"]
        self.spatial_pos_by_pid = None
        self.sub_edge_index_list = None
        self.local_partition_ids = []
        self.local_partitions = []
        self.local_sub_edge_index_for_partition_results = []
        self.local_dup_nodes_per_partition = []
        self.local_spatial_pos_by_pid = []
        self.local_dup_indices = []
        self.local_dup_nodes_per_partition_feature = None

def build_placeholder_struct_info(graph_in_degree, graph_out_degree):
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
    )

def get_rank_source_range(num_nodes: int, rank: int, world_size: int):
    start = (num_nodes * rank) // world_size
    end = (num_nodes * (rank + 1)) // world_size
    return start, end

def gather_ppr_shards(local_ppr: tuple[torch.Tensor, torch.Tensor], rank: int, world_size: int):
    edge_index, edge_value = local_ppr
    if world_size <= 1:
        return edge_index, edge_value

    local_payload = (
        edge_index.detach().cpu(),
        edge_value.detach().cpu(),
    )
    try:
        dist.barrier()
    except Exception as exc:
        raise RuntimeError(
            "Distributed PPR shard synchronization failed before gather. This usually means at least one rank did not finish local APPNP propagation cleanly."
        ) from exc

    gathered_payloads = [None for _ in range(world_size)] if rank == 0 else None
    try:
        dist.gather_object(local_payload, object_gather_list=gathered_payloads, dst=0)
    except Exception as exc:
        raise RuntimeError(
            "Distributed PPR shard gather failed. Check earlier rank-local logs for the first rank that crashed or stalled before entering gather."
        ) from exc

    if rank != 0:
        return None

    edge_index_parts = []
    edge_value_parts = []
    for payload in gathered_payloads:
        if payload is None:
            continue
        shard_edge_index, shard_edge_value = payload
        if shard_edge_value.numel() <= 0:
            continue
        edge_index_parts.append(shard_edge_index)
        edge_value_parts.append(shard_edge_value)

    if not edge_index_parts:
        empty_index = torch.empty((2, 0), dtype=torch.long)
        empty_value = torch.empty((0,), dtype=edge_value.dtype)
        return empty_index, empty_value
    return torch.cat(edge_index_parts, dim=1), torch.cat(edge_value_parts, dim=0)

def build_graph_struct_info(args,N,edge_index,feature,world_size,device,topk=50,n_parts=50,related_nodes_topk_rate=5,connect_prob=0.01,edge_csr_data=None):
    graph_in_degree, graph_out_degree = None, None
    if args.struct_enc == "True":
        if edge_csr_data is not None:
            graph_in_degree, graph_out_degree = _get_node_degrees_from_csr(edge_csr_data, N)
        else:
            graph_in_degree, graph_out_degree = get_node_degrees(edge_index, N)

    distributed_appnp_ppr = world_size > 1 and args.ppr_backend == "appnp"
    if world_size > 1 and not distributed_appnp_ppr and args.rank != 0:
        return build_placeholder_struct_info(graph_in_degree, graph_out_degree)

    source_start, source_end = 0, N
    if distributed_appnp_ppr:
        source_start, source_end = get_rank_source_range(N, args.rank, world_size)
        print(f"[rank {args.rank}] PPR source range: [{source_start}, {source_end})")

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
    local_edge_count = int(local_sorted_ppr_matrix[1].numel())
    print(f"[rank {args.rank}] Local PPR propagation time: {local_ppr_time:.3f}s")
    print(f"[rank {args.rank}] Local PPR edge count: {local_edge_count}")

    sorted_ppr_matrix = local_sorted_ppr_matrix
    ppr_time = local_ppr_time
    gather_time = 0.0
    if distributed_appnp_ppr:
        sync_device(device)
        gather_start = time.time()
        sorted_ppr_matrix = gather_ppr_shards(local_sorted_ppr_matrix, rank=args.rank, world_size=world_size)
        sync_device(device)
        gather_time = time.time() - gather_start
        if args.rank == 0 and sorted_ppr_matrix is not None:
            ppr_time += gather_time
            print(f"[rank 0] Gather PPR shards time: {gather_time:.3f}s")
            print(f"[rank 0] Gathered PPR edge count: {int(sorted_ppr_matrix[1].numel())}")
        if args.rank != 0:
            return build_placeholder_struct_info(graph_in_degree, graph_out_degree)

    graph_edge_index = _ensure_edge_index(edge_index, edge_csr_data)

    isolated_start = time.time()
    sorted_ppr_matrix = add_isolated_connections(sorted_ppr_matrix, graph_edge_index, N, connect_prob=connect_prob)
    isolated_time = time.time() - isolated_start

    adj_build_start = time.time()
    csr_adjacency, eweights, _ = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)
    adj_build_time = time.time() - adj_build_start
    print(f"PPR propagation time: {ppr_time:.3f}s")
    if distributed_appnp_ppr:
        print(f"PPR shard gather time: {gather_time:.3f}s")
    print(f"Isolated-node merge time: {isolated_time:.3f}s")
    print(f"Adjacency build time: {adj_build_time:.3f}s")

    wm = weightMetis_keepParent(
        csr_adjacency=csr_adjacency, 
        eweights=eweights,
        n_parts=n_parts,
        feature=feature,
        edge_index=graph_edge_index,
        related_nodes_topk_rate=related_nodes_topk_rate,
        attn_type=args.attn_type,
        sorted_ppr_matrix=sorted_ppr_matrix)

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
