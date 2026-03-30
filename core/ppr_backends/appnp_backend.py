import torch
from tqdm import tqdm


def _normalize_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _load_csr_graph(csr_data, device, num_nodes: int | None = None):
    if csr_data is None:
        raise ValueError("csr_data must not be None when loading a CSR graph")
    if not isinstance(csr_data, dict):
        raise TypeError(f"Unsupported csr_data type: {type(csr_data)}")
    if "rowptr" not in csr_data or "col" not in csr_data:
        raise KeyError(f"CSR data must contain rowptr and col, got keys={list(csr_data.keys())}")

    rowptr = torch.as_tensor(csr_data["rowptr"], dtype=torch.long, device=device)
    col = torch.as_tensor(csr_data["col"], dtype=torch.long, device=device)
    if rowptr.dim() != 1 or col.dim() != 1:
        raise ValueError("CSR rowptr and col must be 1-D")
    inferred_num_nodes = int(rowptr.numel() - 1)
    if num_nodes is not None and inferred_num_nodes != int(num_nodes):
        raise ValueError(f"CSR num_nodes mismatch: expected {num_nodes}, got {inferred_num_nodes}")
    degree = (rowptr[1:] - rowptr[:-1]).clamp_min(1).to(torch.float32)
    return rowptr, col, degree, inferred_num_nodes


def _build_csr_from_edge_index(edge_index: torch.Tensor, device, num_nodes: int | None = None):
    if edge_index.numel() == 0:
        empty_rowptr = torch.zeros((1,), dtype=torch.long, device=device)
        empty_col = torch.empty((0,), dtype=torch.long, device=device)
        empty_degree = torch.empty((0,), dtype=torch.float32, device=device)
        return empty_rowptr, empty_col, empty_degree, 0

    edge_index = edge_index.to(device)
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    src = edge_index[0].long()
    dst = edge_index[1].long()
    sort_key = src * max(int(num_nodes), 1) + dst
    perm = torch.argsort(sort_key)
    src = src[perm]
    dst = dst[perm]

    counts = torch.bincount(src, minlength=int(num_nodes))
    rowptr = torch.zeros((int(num_nodes) + 1,), dtype=torch.long, device=device)
    rowptr[1:] = torch.cumsum(counts, dim=0)
    degree = counts.clamp_min(1).to(torch.float32)
    return rowptr, dst, degree, int(num_nodes)


def _collect_neighbors(col: torch.Tensor, rowptr: torch.Tensor, node_ids: torch.Tensor):
    neighbor_chunks = []
    lengths = []
    for node in node_ids.tolist():
        start = int(rowptr[node].item())
        end = int(rowptr[node + 1].item())
        neighbors = col[start:end]
        neighbor_chunks.append(neighbors)
        lengths.append(neighbors.numel())
    return neighbor_chunks, lengths


def _propagate_single_seed(
    seed: int,
    current_nodes: torch.Tensor,
    current_values: torch.Tensor,
    rowptr: torch.Tensor,
    col: torch.Tensor,
    degree: torch.Tensor,
    alpha: float,
    iter_topk: int,
    device,
):
    if current_nodes.numel() == 0:
        return (
            torch.tensor([seed], dtype=torch.long, device=device),
            torch.tensor([alpha], dtype=torch.float32, device=device),
        )

    neighbor_chunks, lengths = _collect_neighbors(col, rowptr, current_nodes)
    non_empty_chunks = [chunk for chunk in neighbor_chunks if chunk.numel() > 0]

    all_nodes_parts = []
    all_values_parts = []
    if non_empty_chunks:
        valid_lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
        repeated_values = ((1.0 - alpha) * current_values / degree[current_nodes]).repeat_interleave(valid_lengths)
        all_nodes_parts.append(torch.cat(non_empty_chunks, dim=0))
        all_values_parts.append(repeated_values)

    all_nodes_parts.append(torch.tensor([seed], dtype=torch.long, device=device))
    all_values_parts.append(torch.tensor([alpha], dtype=torch.float32, device=device))

    all_nodes = torch.cat(all_nodes_parts, dim=0)
    all_values = torch.cat(all_values_parts, dim=0)

    unique_nodes, inverse = torch.unique(all_nodes, sorted=False, return_inverse=True)
    aggregated_values = torch.zeros(unique_nodes.numel(), dtype=torch.float32, device=device)
    aggregated_values.scatter_add_(0, inverse, all_values)

    keep_k = min(int(iter_topk), int(unique_nodes.numel()))
    if keep_k <= 0:
        return (
            torch.tensor([seed], dtype=torch.long, device=device),
            torch.tensor([alpha], dtype=torch.float32, device=device),
        )

    top_values, top_idx = torch.topk(aggregated_values, k=keep_k, largest=True, sorted=True)
    top_nodes = unique_nodes[top_idx]
    return top_nodes, top_values


def personal_pagerank_appnp(
    edge_index,
    alpha,
    topk=100,
    num_iterations: int = 10,
    batch_size: int = 8,
    device="cuda",
    csr_data=None,
    num_nodes: int | None = None,
    iter_topk: int | None = None,
) -> tuple:
    device = _normalize_device(device)
    batch_size = max(1, int(batch_size))
    num_iterations = max(1, int(num_iterations))

    if csr_data is not None:
        rowptr, col, degree, num_nodes = _load_csr_graph(csr_data, device=device, num_nodes=num_nodes)
    else:
        if edge_index is None:
            raise ValueError("edge_index must be provided when csr_data is None")
        rowptr, col, degree, num_nodes = _build_csr_from_edge_index(edge_index, device=device, num_nodes=num_nodes)

    if num_nodes == 0:
        empty_index = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_value = torch.empty((0,), dtype=torch.float32, device=device)
        return empty_index, empty_value

    topk = min(int(topk), num_nodes)
    iter_topk = topk if iter_topk is None else min(max(1, int(iter_topk)), num_nodes)

    edge_index_batches = []
    edge_value_batches = []

    for start in tqdm(range(0, num_nodes, batch_size), desc="appnp ppr"):
        end = min(start + batch_size, num_nodes)
        seed_nodes = torch.arange(start, end, device=device, dtype=torch.long)

        current_rowptr = torch.arange(0, seed_nodes.numel() + 1, dtype=torch.long, device=device)
        current_col = seed_nodes.clone()
        current_values = torch.ones(seed_nodes.numel(), dtype=torch.float32, device=device)

        for _ in range(num_iterations):
            next_rowptr = [0]
            next_cols = []
            next_values = []
            for local_seed_idx, seed in enumerate(seed_nodes.tolist()):
                row_start = int(current_rowptr[local_seed_idx].item())
                row_end = int(current_rowptr[local_seed_idx + 1].item())
                top_nodes, top_values = _propagate_single_seed(
                    seed=seed,
                    current_nodes=current_col[row_start:row_end],
                    current_values=current_values[row_start:row_end],
                    rowptr=rowptr,
                    col=col,
                    degree=degree,
                    alpha=alpha,
                    iter_topk=iter_topk,
                    device=device,
                )
                next_cols.append(top_nodes)
                next_values.append(top_values)
                next_rowptr.append(next_rowptr[-1] + int(top_nodes.numel()))

            current_rowptr = torch.tensor(next_rowptr, dtype=torch.long, device=device)
            current_col = torch.cat(next_cols, dim=0) if next_cols else torch.empty((0,), dtype=torch.long, device=device)
            current_values = torch.cat(next_values, dim=0) if next_values else torch.empty((0,), dtype=torch.float32, device=device)

        final_keep = min(topk, iter_topk)
        if final_keep < iter_topk:
            final_rowptr = [0]
            final_cols = []
            final_values = []
            for local_seed_idx in range(seed_nodes.numel()):
                row_start = int(current_rowptr[local_seed_idx].item())
                row_end = int(current_rowptr[local_seed_idx + 1].item())
                row_nodes = current_col[row_start:row_end]
                row_values = current_values[row_start:row_end]
                if row_values.numel() > final_keep:
                    row_values, top_idx = torch.topk(row_values, k=final_keep, largest=True, sorted=True)
                    row_nodes = row_nodes[top_idx]
                final_cols.append(row_nodes)
                final_values.append(row_values)
                final_rowptr.append(final_rowptr[-1] + int(row_nodes.numel()))
            current_rowptr = torch.tensor(final_rowptr, dtype=torch.long, device=device)
            current_col = torch.cat(final_cols, dim=0) if final_cols else torch.empty((0,), dtype=torch.long, device=device)
            current_values = torch.cat(final_values, dim=0) if final_values else torch.empty((0,), dtype=torch.float32, device=device)

        row_lengths = current_rowptr[1:] - current_rowptr[:-1]
        batch_sources = seed_nodes.repeat_interleave(row_lengths)
        edge_index_batches.append(torch.stack([batch_sources, current_col], dim=0))
        edge_value_batches.append(current_values)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(edge_index_batches, dim=1), torch.cat(edge_value_batches, dim=0)
