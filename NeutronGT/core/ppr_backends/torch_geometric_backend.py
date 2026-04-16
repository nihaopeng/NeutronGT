import torch
from torch_geometric.utils import ppr


def personal_pagerank_torch_geometric(edge_index, alpha, topk=100, eps: float = 1e-6, device="cuda") -> tuple:
    edge_indices, edge_values = ppr.get_ppr(
        edge_index,
        alpha=alpha,
        eps=eps,
    )
    edge_indices = edge_indices.to(device)
    edge_values = edge_values.to(device)
    source_nodes = edge_indices[0]
    unique_sources = torch.unique(source_nodes)
    topk_indices_list = []
    topk_values_list = []
    for src in unique_sources:
        mask = source_nodes == src
        src_edges = edge_indices[:, mask]
        src_values = edge_values[mask]
        if src_values.shape[0] <= topk:
            topk_indices_list.append(src_edges)
            topk_values_list.append(src_values)
        else:
            topk_idx = torch.topk(src_values, topk, largest=True)[1]
            topk_indices_list.append(src_edges[:, topk_idx])
            topk_values_list.append(src_values[topk_idx])
    topk_indices = torch.cat(topk_indices_list, dim=1)
    topk_values = torch.cat(topk_values_list)
    return topk_indices, topk_values
