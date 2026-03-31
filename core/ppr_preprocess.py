import os
import networkx as nx
import torch
from tqdm import tqdm
import numpy as np
import pymetis

from core.ppr_backends import personal_pagerank_appnp, personal_pagerank_torch_geometric


def metis_partition(csr_adjacency:pymetis.CSRAdjacency,eweights:list[list],n_parts):
    # 将训练索引转换为集合，用于快速查找
    try:
        n_cuts, membership = pymetis.part_graph(
            nparts=n_parts,
            adjacency=csr_adjacency,
            eweights=eweights
        )
    except Exception as e:
        print(f"Metis failed: {e}")
        raise
    partitions = [[] for _ in range(n_parts)]
    for node_idx, part_id in enumerate(membership):
        partitions[part_id].append(node_idx)
    filtered_partitions = []
    for part in partitions:
        filtered_partitions.append(torch.tensor(part, dtype=torch.long))
    return filtered_partitions


def build_adj_fromat(sorted_ppr_matrix):
    """FROM QWEN"""
    print("======start adj format building===========")
    edge_index, ppr_val = sorted_ppr_matrix
    assert edge_index.shape[0] == 2
    if edge_index.numel() == 0:
        csr_adj = pymetis.CSRAdjacency(adj_starts=[0], adjacent=[])
        return csr_adj, [], None

    num_nodes = int(edge_index.max().item()) + 1
    src, dst = edge_index[0].long(), edge_index[1].long()
    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    edge_key = u * num_nodes + v
    unique_edge_keys, inverse_indices = torch.unique(edge_key, sorted=True, return_inverse=True)
    summed_ppr = torch.zeros(unique_edge_keys.numel(), dtype=ppr_val.dtype, device=ppr_val.device)
    summed_ppr.scatter_add_(0, inverse_indices, ppr_val)
    unique_u = torch.div(unique_edge_keys, num_nodes, rounding_mode="floor")
    unique_v = unique_edge_keys % num_nodes
    weights = (summed_ppr * 1000).clamp_min(1).to(torch.int32)

    print("======构建无向连接===========")
    u_all = torch.cat([unique_u, unique_v], dim=0)
    v_all = torch.cat([unique_v, unique_u], dim=0)
    weights_all = torch.cat([weights, weights], dim=0)
    sort_idx = torch.argsort(u_all)
    u_all = u_all[sort_idx]
    v_all = v_all[sort_idx]
    weights_all = weights_all[sort_idx]

    print("======csr format building===========")
    degrees = torch.bincount(u_all, minlength=num_nodes).to(torch.int32)
    xadj = torch.zeros(num_nodes + 1, dtype=torch.int32, device=u_all.device)
    xadj[1:] = torch.cumsum(degrees, dim=0)

    xadj_np = xadj.cpu().numpy()
    adjncy_np = v_all.to(torch.int32).cpu().numpy()
    eweights_np = weights_all.cpu().numpy()
    assert len(adjncy_np) == len(eweights_np)
    assert int(xadj_np[-1]) == len(adjncy_np)
    csr_adj = pymetis.CSRAdjacency(
        adj_starts=xadj_np.tolist(),
        adjacent=adjncy_np.tolist()
    )
    return csr_adj, eweights_np.tolist(), None


def ppr_partition(sorted_ppr_matrix:list[torch.tensor,torch.tensor],flatten_train_idx,num_set:int):
    train_set = set(flatten_train_idx)
    partitioned_results = []
    print(f"num_of_ppr:{len(sorted_ppr_matrix)}")
    for start_idx in tqdm(range(0, len(sorted_ppr_matrix), num_set), desc="ppr partition"):
        end_idx = min(start_idx + num_set, len(sorted_ppr_matrix))
        node_set = set()
        for j in range(start_idx, end_idx):
            if sorted_ppr_matrix.get(j, None) is None:
                break
            ppr_nodes = [item[0] for item in sorted_ppr_matrix[j]]
            node_set |= set(ppr_nodes)
        if not node_set:
            print("None type found!")
            continue
        partitioned_results.append(list(node_set))
    filtered_partitions = []
    for partition in partitioned_results:
        filtered_partitions.append(torch.tensor(partition, dtype=torch.long))
    return filtered_partitions


def add_isolated_connections(
    ppr_result,
    original_edge_index: torch.Tensor,
    num_nodes: int,
    connect_prob: float = 0.01,
    ppr_fill_value: float = 0.001,
    device="cuda"
) -> tuple:
    """FROM QWEN
    向 PPR 结果中添加孤立节点与其他节点的随机连接。
    """
    edge_index, edge_values = ppr_result
    edge_index = edge_index.to(device)
    edge_values = edge_values.to(device)
    original_edge_index = original_edge_index.to(device)
    ppr_appeared_nodes = torch.unique(edge_index)
    original_appeared_nodes = torch.unique(original_edge_index)
    appeared_nodes = torch.unique(torch.cat([ppr_appeared_nodes, original_appeared_nodes], dim=0))
    all_nodes = torch.arange(num_nodes, device=device)
    is_isolated = ~torch.isin(all_nodes, appeared_nodes)
    isolated_nodes = all_nodes[is_isolated]
    if isolated_nodes.numel() == 0:
        return edge_index, edge_values
    non_isolated_mask = ~is_isolated
    non_isolated_nodes = all_nodes[non_isolated_mask]
    if non_isolated_nodes.numel() == 0:
        return edge_index, edge_values
    rand_probs = torch.rand(len(isolated_nodes), len(non_isolated_nodes), device=device)
    connect_mask = rand_probs < connect_prob
    i_idx, j_idx = torch.where(connect_mask)
    if i_idx.numel() == 0:
        return edge_index, edge_values
    srcs = isolated_nodes[i_idx]
    dsts = non_isolated_nodes[j_idx]
    new_edges = torch.stack([srcs, dsts], dim=0)
    new_values = torch.full((new_edges.shape[1],), ppr_fill_value, device=device, dtype=edge_values.dtype)
    final_edge_index = torch.cat([edge_index, new_edges], dim=1)
    final_edge_values = torch.cat([edge_values, new_values], dim=0)
    return final_edge_index, final_edge_values


def personal_pagerank(
    edge_index,
    alpha,
    topk=100,
    max_iter: int = 100,
    device="cuda",
    backend: str = "torch_geometric",
    num_iterations: int | None = None,
    batch_size: int = 8,
    eps: float = 1e-6,
    csr_data=None,
    num_nodes: int | None = None,
    iter_topk: int | None = None,
    source_start: int | None = None,
    source_end: int | None = None,
) -> tuple:
    """为所有节点计算个性化PageRank。"""
    if backend == "torch_geometric":
        if source_start is not None or source_end is not None:
            raise NotImplementedError("torch_geometric backend does not support distributed source sharding")
        return personal_pagerank_torch_geometric(edge_index, alpha, topk=topk, eps=eps, device=device)
    if backend == "appnp":
        if num_iterations is None:
            num_iterations = max_iter
        return personal_pagerank_appnp(
            edge_index,
            alpha,
            topk=topk,
            num_iterations=num_iterations,
            batch_size=batch_size,
            device=device,
            csr_data=csr_data,
            num_nodes=num_nodes,
            iter_topk=iter_topk,
            source_start=source_start,
            source_end=source_end,
        )
    raise ValueError(f"Unsupported PPR backend: {backend}")


if __name__ == "__main__":
    dataset_dir = "./dataset"
    dataset_name = "cora"
    feature = torch.load(os.path.join(dataset_dir, dataset_name, "x.pt"))
    y = torch.load(os.path.join(dataset_dir, dataset_name, "y.pt"))
    edge_index = torch.load(os.path.join(dataset_dir, dataset_name, "edge_index.pt"))
    N = feature.shape[0]

    sorted_ppr_matrix = personal_pagerank(edge_index, alpha=0.85, topk=100)
    csr_adjacency, eweights, _ = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)
    partitioned_results = metis_partition(csr_adjacency, eweights, 10)
    print(f"idx0:{partitioned_results[0]},num:{len(partitioned_results[0])}")
