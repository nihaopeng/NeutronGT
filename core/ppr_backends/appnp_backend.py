import time

import torch
from tqdm import tqdm

try:
    import dgl.sparse as dglsp
except ImportError:  # pragma: no cover - runtime availability depends on the environment
    dglsp = None


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


def _rowptr_to_rows(rowptr: torch.Tensor):
    lengths = rowptr[1:] - rowptr[:-1]
    if lengths.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=rowptr.device)
    return torch.arange(lengths.numel(), dtype=torch.long, device=rowptr.device).repeat_interleave(lengths)


def _aggregate_sparse_entries(rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, num_nodes: int):
    if rows.numel() == 0:
        return rows, cols, values
    key = rows.to(torch.long) * int(num_nodes) + cols.to(torch.long)
    unique_key, inverse = torch.unique(key, sorted=False, return_inverse=True)
    aggregated_values = torch.zeros(unique_key.numel(), dtype=values.dtype, device=values.device)
    aggregated_values.scatter_add_(0, inverse, values)
    aggregated_rows = torch.div(unique_key, int(num_nodes), rounding_mode="floor")
    aggregated_cols = unique_key % int(num_nodes)
    return aggregated_rows, aggregated_cols, aggregated_values


def _build_rowptr_from_rows(rows: torch.Tensor, num_rows: int):
    row_counts = torch.bincount(rows, minlength=num_rows)
    rowptr = torch.zeros((num_rows + 1,), dtype=torch.long, device=rows.device)
    rowptr[1:] = torch.cumsum(row_counts, dim=0)
    return rowptr


def _coo_to_csr_state(rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, num_rows: int):
    if rows.numel() == 0:
        empty_rowptr = torch.zeros((num_rows + 1,), dtype=torch.long, device=values.device)
        empty_cols = torch.empty((0,), dtype=torch.long, device=values.device)
        empty_values = torch.empty((0,), dtype=values.dtype, device=values.device)
        return empty_rowptr, empty_cols, empty_values

    sort_idx = torch.argsort(rows)
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    values = values[sort_idx]
    rowptr = _build_rowptr_from_rows(rows, num_rows)
    return rowptr, cols, values


def _segment_topk_from_csr(rowptr: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, keep_k: int | None):
    num_rows = int(rowptr.numel() - 1)
    if cols.numel() == 0 or keep_k is None:
        return rowptr, cols, values

    keep_k = int(keep_k)
    if keep_k <= 0:
        return rowptr, cols, values

    row_lengths = rowptr[1:] - rowptr[:-1]
    if row_lengths.numel() == 0 or int(row_lengths.max().item()) <= keep_k:
        return rowptr, cols, values

    kept_counts = torch.clamp(row_lengths, max=keep_k)
    next_rowptr = torch.zeros((num_rows + 1,), dtype=torch.long, device=values.device)
    next_rowptr[1:] = torch.cumsum(kept_counts, dim=0)

    total_kept = int(next_rowptr[-1].item())
    topk_cols = torch.empty((total_kept,), dtype=torch.long, device=values.device)
    topk_values = torch.empty((total_kept,), dtype=values.dtype, device=values.device)

    rowptr_list = rowptr.tolist()
    next_rowptr_list = next_rowptr.tolist()
    nonzero_rows = torch.nonzero(row_lengths, as_tuple=False).flatten().tolist()
    for row_id in nonzero_rows:
        row_start = rowptr_list[row_id]
        row_end = rowptr_list[row_id + 1]
        out_start = next_rowptr_list[row_id]
        out_end = next_rowptr_list[row_id + 1]

        row_cols = cols[row_start:row_end]
        row_values = values[row_start:row_end]
        if row_values.numel() > keep_k:
            row_values, top_idx = torch.topk(row_values, k=keep_k, largest=True, sorted=True)
            row_cols = row_cols[top_idx]
        topk_cols[out_start:out_end] = row_cols
        topk_values[out_start:out_end] = row_values

    return next_rowptr, topk_cols, topk_values


def _segment_topk(rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, num_rows: int, keep_k: int | None):
    rowptr, cols, values = _coo_to_csr_state(rows, cols, values, num_rows)
    return _segment_topk_from_csr(rowptr, cols, values, keep_k)


def _state_to_dgl_sparse(rowptr: torch.Tensor, col: torch.Tensor, values: torch.Tensor, num_rows: int, num_cols: int):
    if dglsp is None:
        raise RuntimeError("dgl.sparse is not available")
    return dglsp.from_csr(rowptr, col, values, shape=(num_rows, num_cols))


def _dgl_sparse_to_state(spmat, num_rows: int, keep_k: int | None):
    rowptr, col, value_idx = spmat.csr()
    values = spmat.val
    if value_idx is not None:
        values = values[value_idx]
    return _segment_topk_from_csr(rowptr, col, values, keep_k)


def _build_transition_dgl_sparse(graph_rowptr: torch.Tensor, graph_col: torch.Tensor, degree: torch.Tensor, num_nodes: int):
    if dglsp is None:
        raise RuntimeError("dgl.sparse is not available")
    src = _rowptr_to_rows(graph_rowptr)
    values = (1.0 / degree[src]).to(torch.float32) if src.numel() > 0 else torch.empty((0,), dtype=torch.float32, device=graph_rowptr.device)
    return dglsp.from_csr(graph_rowptr, graph_col, values, shape=(num_nodes, num_nodes))


def _dgl_spgemm_iteration(state_rowptr, state_col, state_values, transition_sparse, teleport_rows, teleport_cols, teleport_values, num_rows, num_nodes, alpha, iter_topk, timing_stats=None):
    step_start = time.perf_counter()
    state_sparse = _state_to_dgl_sparse(state_rowptr, state_col, state_values, num_rows, num_nodes)
    if timing_stats is not None:
        timing_stats["state_to_sparse"] += time.perf_counter() - step_start

    step_start = time.perf_counter()
    propagated_sparse = dglsp.spspmm(state_sparse, transition_sparse)
    propagated_sparse = dglsp.val_like(propagated_sparse, propagated_sparse.val * (1.0 - alpha))
    if timing_stats is not None:
        timing_stats["spmm"] += time.perf_counter() - step_start

    step_start = time.perf_counter()
    prop_rowptr, prop_col, value_idx = propagated_sparse.csr()
    prop_values = propagated_sparse.val
    if value_idx is not None:
        prop_values = prop_values[value_idx]
    prop_rows = _rowptr_to_rows(prop_rowptr)
    merged_rows = torch.cat([prop_rows, teleport_rows], dim=0)
    merged_cols = torch.cat([prop_col, teleport_cols], dim=0)
    merged_values = torch.cat([prop_values, teleport_values], dim=0)
    merged_rows, merged_cols, merged_values = _aggregate_sparse_entries(
        merged_rows,
        merged_cols,
        merged_values,
        num_nodes,
    )
    if timing_stats is not None:
        timing_stats["merge"] += time.perf_counter() - step_start

    step_start = time.perf_counter()
    if iter_topk is None:
        next_state = _coo_to_csr_state(merged_rows, merged_cols, merged_values, num_rows)
    else:
        next_state = _segment_topk(merged_rows, merged_cols, merged_values, num_rows, iter_topk)
    if timing_stats is not None:
        timing_stats["segment_topk"] += time.perf_counter() - step_start
    return next_state


def _expand_batch_neighbors(batch_rows, batch_nodes, batch_values, graph_rowptr, graph_col, degree, alpha):
    if batch_nodes.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=graph_rowptr.device)
        empty_values = torch.empty((0,), dtype=torch.float32, device=graph_rowptr.device)
        return empty, empty, empty_values

    lengths = graph_rowptr[batch_nodes + 1] - graph_rowptr[batch_nodes]
    valid_mask = lengths > 0
    if not valid_mask.any():
        empty = torch.empty((0,), dtype=torch.long, device=graph_rowptr.device)
        empty_values = torch.empty((0,), dtype=torch.float32, device=graph_rowptr.device)
        return empty, empty, empty_values

    batch_rows = batch_rows[valid_mask]
    batch_nodes = batch_nodes[valid_mask]
    batch_values = batch_values[valid_mask]
    lengths = lengths[valid_mask]
    total_nnz = int(lengths.sum().item())

    starts = graph_rowptr[batch_nodes]
    segment_offsets = torch.cumsum(lengths, dim=0) - lengths
    repeated_starts = starts.repeat_interleave(lengths)
    repeated_offsets = segment_offsets.repeat_interleave(lengths)
    local_offsets = torch.arange(total_nnz, dtype=torch.long, device=graph_rowptr.device) - repeated_offsets
    gather_positions = repeated_starts + local_offsets

    expanded_rows = batch_rows.repeat_interleave(lengths)
    expanded_cols = graph_col[gather_positions]
    expanded_values = ((1.0 - alpha) * batch_values / degree[batch_nodes]).repeat_interleave(lengths)
    return expanded_rows, expanded_cols, expanded_values


def _fallback_iteration(state_rowptr, state_col, state_values, graph_rowptr, graph_col, degree, teleport_rows, teleport_cols, teleport_values, num_rows, num_nodes, alpha, iter_topk):
    state_rows = _rowptr_to_rows(state_rowptr)
    propagated_rows, propagated_cols, propagated_values = _expand_batch_neighbors(
        state_rows,
        state_col,
        state_values,
        graph_rowptr,
        graph_col,
        degree,
        alpha,
    )
    merged_rows = torch.cat([propagated_rows, teleport_rows], dim=0)
    merged_cols = torch.cat([propagated_cols, teleport_cols], dim=0)
    merged_values = torch.cat([propagated_values, teleport_values], dim=0)
    merged_rows, merged_cols, merged_values = _aggregate_sparse_entries(
        merged_rows,
        merged_cols,
        merged_values,
        num_nodes,
    )
    return _segment_topk(merged_rows, merged_cols, merged_values, num_rows, iter_topk)


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
    source_start: int | None = None,
    source_end: int | None = None,
) -> tuple:
    device = _normalize_device(device)
    batch_size = max(1, int(batch_size))
    num_iterations = max(1, int(num_iterations))

    if csr_data is not None:
        graph_rowptr, graph_col, degree, num_nodes = _load_csr_graph(csr_data, device=device, num_nodes=num_nodes)
    else:
        if edge_index is None:
            raise ValueError("edge_index must be provided when csr_data is None")
        graph_rowptr, graph_col, degree, num_nodes = _build_csr_from_edge_index(edge_index, device=device, num_nodes=num_nodes)

    if num_nodes == 0:
        empty_index = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_value = torch.empty((0,), dtype=torch.float32, device=device)
        return empty_index, empty_value

    source_start = 0 if source_start is None else max(0, min(int(source_start), num_nodes))
    source_end = num_nodes if source_end is None else max(source_start, min(int(source_end), num_nodes))
    if source_start >= source_end:
        empty_index = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_value = torch.empty((0,), dtype=torch.float32, device=device)
        return empty_index, empty_value

    topk = min(int(topk), num_nodes)
    iter_topk = topk if iter_topk is None else int(iter_topk)
    iter_topk = None if iter_topk <= 0 else min(iter_topk, num_nodes)
    transition_sparse = _build_transition_dgl_sparse(graph_rowptr, graph_col, degree, num_nodes) if dglsp is not None else None
    use_dgl_spgemm = transition_sparse is not None
    logged_fallback = False
    timing_stats = {
        "state_to_sparse": 0.0,
        "spmm": 0.0,
        "merge": 0.0,
        "segment_topk": 0.0,
        "fallback": 0.0,
        "final_topk": 0.0,
    }

    edge_index_batches = []
    edge_value_batches = []

    for start in tqdm(range(source_start, source_end, batch_size), desc="appnp ppr"):
        end = min(start + batch_size, source_end)
        seed_nodes = torch.arange(start, end, dtype=torch.long, device=device)
        num_rows = seed_nodes.numel()

        state_rowptr = torch.arange(0, num_rows + 1, dtype=torch.long, device=device)
        state_col = seed_nodes.clone()
        state_values = torch.ones(num_rows, dtype=torch.float32, device=device)

        teleport_rows = torch.arange(num_rows, dtype=torch.long, device=device)
        teleport_cols = seed_nodes
        teleport_values = torch.full((num_rows,), alpha, dtype=torch.float32, device=device)

        for _ in range(num_iterations):
            if use_dgl_spgemm:
                try:
                    state_rowptr, state_col, state_values = _dgl_spgemm_iteration(
                        state_rowptr,
                        state_col,
                        state_values,
                        transition_sparse,
                        teleport_rows,
                        teleport_cols,
                        teleport_values,
                        num_rows,
                        num_nodes,
                        alpha,
                        iter_topk,
                        timing_stats=timing_stats,
                    )
                    continue
                except RuntimeError as exc:
                    use_dgl_spgemm = False
                    if not logged_fallback:
                        print(f"[APPNP] fallback triggered, switch to manual sparse propagation: {exc}")
                        logged_fallback = True
            fallback_start = time.perf_counter()
            state_rowptr, state_col, state_values = _fallback_iteration(
                state_rowptr,
                state_col,
                state_values,
                graph_rowptr,
                graph_col,
                degree,
                teleport_rows,
                teleport_cols,
                teleport_values,
                num_rows,
                num_nodes,
                alpha,
                iter_topk,
            )
            timing_stats["fallback"] += time.perf_counter() - fallback_start

        final_keep = topk if iter_topk is None else min(topk, iter_topk)
        if final_keep is not None:
            final_topk_start = time.perf_counter()
            state_rowptr, state_col, state_values = _segment_topk_from_csr(
                state_rowptr,
                state_col,
                state_values,
                final_keep,
            )
            timing_stats["final_topk"] += time.perf_counter() - final_topk_start

        row_lengths = state_rowptr[1:] - state_rowptr[:-1]
        batch_sources = seed_nodes.repeat_interleave(row_lengths)
        edge_index_batches.append(torch.stack([batch_sources, state_col], dim=0))
        edge_value_batches.append(state_values)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(edge_index_batches, dim=1), torch.cat(edge_value_batches, dim=0)
