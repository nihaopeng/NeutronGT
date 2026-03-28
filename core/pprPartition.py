import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pymetis
import torch
from torch_geometric.utils import ppr
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORA_BIN = REPO_ROOT / "third_party" / "fora" / "build" / "fora"
DEFAULT_FORA_WORK_DIR = REPO_ROOT / "third_party" / "fora" / "data"


@dataclass
class GraphTopology:
    num_nodes: int
    source_format: str
    edge_index: Optional[torch.Tensor] = None
    rowptr: Optional[torch.Tensor] = None
    col: Optional[torch.Tensor] = None

    def get_edge_index(self) -> torch.Tensor:
        if self.edge_index is None:
            if self.rowptr is None or self.col is None:
                raise ValueError("Cannot reconstruct edge_index without CSR data")
            self.edge_index = csr_to_edge_index(self.rowptr, self.col)
        return self.edge_index


def _validate_csr(rowptr: torch.Tensor, col: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rowptr = rowptr.detach().cpu().long()
    col = col.detach().cpu().long()
    if rowptr.ndim != 1 or col.ndim != 1:
        raise ValueError(f"CSR tensors must be 1-D, got rowptr={tuple(rowptr.shape)}, col={tuple(col.shape)}")
    if rowptr.numel() == 0:
        raise ValueError("CSR rowptr must not be empty")
    if int(rowptr[0].item()) != 0:
        raise ValueError("CSR rowptr[0] must be 0")
    if rowptr[1:].numel() > 0 and torch.any(rowptr[1:] < rowptr[:-1]):
        raise ValueError("CSR rowptr must be nondecreasing")
    if int(rowptr[-1].item()) != int(col.numel()):
        raise ValueError(f"CSR rowptr[-1] must equal len(col), got {int(rowptr[-1].item())} vs {int(col.numel())}")
    return rowptr, col


def load_graph_topology(dataset_dir, dataset_name, num_nodes: Optional[int] = None) -> GraphTopology:
    dataset_path = Path(dataset_dir) / dataset_name
    csr_path = dataset_path / "edge_index_csr.pt"
    edge_index_path = dataset_path / "edge_index.pt"
    if csr_path.exists():
        payload = torch.load(csr_path, map_location="cpu")
        if isinstance(payload, dict):
            rowptr = payload.get("rowptr")
            col = payload.get("col")
        elif isinstance(payload, (tuple, list)) and len(payload) == 2:
            rowptr, col = payload
        else:
            raise ValueError(f"Unsupported CSR payload format in {csr_path}")
        if rowptr is None or col is None:
            raise ValueError(f"CSR payload in {csr_path} must contain rowptr and col")
        rowptr, col = _validate_csr(rowptr, col)
        inferred_num_nodes = int(rowptr.numel()) - 1
        if num_nodes is not None and inferred_num_nodes != int(num_nodes):
            raise ValueError(f"CSR num_nodes mismatch: inferred {inferred_num_nodes}, expected {int(num_nodes)}")
        return GraphTopology(
            num_nodes=inferred_num_nodes,
            source_format="csr",
            edge_index=None,
            rowptr=rowptr,
            col=col,
        )
    if not edge_index_path.exists():
        raise FileNotFoundError(f"Neither {csr_path} nor {edge_index_path} exists")
    edge_index = torch.load(edge_index_path, map_location="cpu")
    edge_index = edge_index.detach().cpu().long()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    inferred_num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else int(num_nodes or 0)
    if num_nodes is not None and inferred_num_nodes != int(num_nodes):
        raise ValueError(f"edge_index num_nodes mismatch: inferred {inferred_num_nodes}, expected {int(num_nodes)}")
    return GraphTopology(
        num_nodes=inferred_num_nodes,
        source_format="coo",
        edge_index=edge_index,
        rowptr=None,
        col=None,
    )


def csr_to_edge_index(rowptr: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    rowptr, col = _validate_csr(rowptr, col)
    num_nodes = int(rowptr.numel()) - 1
    if num_nodes == 0 or col.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    counts = rowptr[1:] - rowptr[:-1]
    src = torch.repeat_interleave(torch.arange(num_nodes, dtype=torch.long), counts)
    return torch.stack([src, col], dim=0)


def _format_alpha(alpha: float) -> str:
    return str(alpha).replace('.', 'p')


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_cache_path(dataset_dir, dataset_name, ppr_cache_dir, backend: str, topk: int, alpha: float) -> Path:
    if ppr_cache_dir is not None:
        cache_root = Path(ppr_cache_dir)
    elif dataset_dir is not None and dataset_name is not None:
        cache_root = Path(dataset_dir) / dataset_name
    else:
        cache_root = REPO_ROOT / 'cache'
    _ensure_dir(cache_root)
    return cache_root / f'ppr_{backend}_topk{topk}_alpha{_format_alpha(alpha)}.pt'


def export_graph_for_fora(edge_index: torch.Tensor, num_nodes: int, output_dir) -> tuple[Path, Path, Path]:
    output_dir = _ensure_dir(Path(output_dir))
    graph_path = output_dir / 'graph.txt'
    attr_path = output_dir / 'attribute.txt'
    ssquery_path = output_dir / 'ssquery.txt'

    edge_index = edge_index.detach().cpu().long()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f'edge_index must have shape [2, E], got {tuple(edge_index.shape)}')

    edges = edge_index.t().contiguous()
    if edges.numel() == 0:
        unique_edges = edges
    else:
        unique_edges = torch.unique(edges, dim=0)
        unique_edges = unique_edges[unique_edges[:, 0] != unique_edges[:, 1]]

    graph_path.write_text(''.join(f'{int(src)} {int(dst)}\n' for src, dst in unique_edges.tolist()), encoding='utf-8')
    attr_path.write_text(f'n={int(num_nodes)}\nm={int(unique_edges.shape[0])}\n', encoding='utf-8')
    ssquery_path.write_text(''.join(f'{i}\n' for i in range(int(num_nodes))), encoding='utf-8')
    return graph_path, attr_path, ssquery_path


def export_graph_for_fora_from_csr(rowptr: torch.Tensor, col: torch.Tensor, num_nodes: int, output_dir) -> tuple[Path, Path, Path]:
    output_dir = _ensure_dir(Path(output_dir))
    graph_path = output_dir / 'graph.txt'
    attr_path = output_dir / 'attribute.txt'
    ssquery_path = output_dir / 'ssquery.txt'

    rowptr, col = _validate_csr(rowptr, col)
    edge_count = 0
    with graph_path.open('w', encoding='utf-8') as graph_file:
        for src in range(int(num_nodes)):
            start = int(rowptr[src].item())
            end = int(rowptr[src + 1].item())
            if start >= end:
                continue
            for dst in col[start:end].tolist():
                dst = int(dst)
                if dst == src:
                    continue
                graph_file.write(f'{src} {dst}\n')
                edge_count += 1
    attr_path.write_text(f'n={int(num_nodes)}\nm={int(edge_count)}\n', encoding='utf-8')
    with ssquery_path.open('w', encoding='utf-8') as query_file:
        for node in range(int(num_nodes)):
            query_file.write(f'{node}\n')
    return graph_path, attr_path, ssquery_path


def build_fora_if_needed(fora_bin=None) -> Path:
    resolved = (Path(fora_bin).expanduser().resolve() if fora_bin is not None else DEFAULT_FORA_BIN.resolve())
    if resolved.exists():
        return resolved
    build_script = REPO_ROOT / 'scripts' / 'build_fora.sh'
    if not build_script.exists():
        raise FileNotFoundError(f'FORA binary not found at {resolved} and build script is missing: {build_script}')
    subprocess.run([str(build_script)], check=True, cwd=REPO_ROOT)
    if not resolved.exists():
        raise FileNotFoundError(f'FORA build finished but binary still not found at {resolved}')
    return resolved


def _parse_fora_dump(dump_path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    src_list = []
    dst_list = []
    val_list = []
    with dump_path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 3:
                continue
            src, dst, score = parts
            src_list.append(int(src))
            dst_list.append(int(dst))
            val_list.append(float(score))
    if not src_list:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.float32, device=device)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    edge_values = torch.tensor(val_list, dtype=torch.float32, device=device)
    return edge_index, edge_values


def _run_fora_topk(fora_bin: Path, prefix_dir: Path, dataset_name: str, alpha: float, topk: int, epsilon: float, query_nodes: list[int], dump_path: Path):
    graph_dir = prefix_dir / dataset_name
    ssquery_path = graph_dir / 'ssquery.txt'
    ssquery_path.write_text(''.join(f'{node}\n' for node in query_nodes), encoding='utf-8')
    cmd = [
        str(fora_bin), 'topk',
        '--algo', 'fora',
        '--prefix', str(prefix_dir) + os.sep,
        '--dataset', dataset_name,
        '--epsilon', str(epsilon),
        '--alpha', str(alpha),
        '--query_size', str(len(query_nodes)),
        '--k', str(topk),
        '--dump_topk_path', str(dump_path),
    ]
    subprocess.run(cmd, check=True, cwd=fora_bin.parent)


def personal_pagerank_torchgeo(edge_index, alpha, topk=100, max_iter: int = 100, device='cuda', rowptr=None, col=None) -> tuple:
    if edge_index is None:
        if rowptr is None or col is None:
            raise ValueError('torchgeo backend requires edge_index or CSR inputs')
        edge_index = csr_to_edge_index(rowptr, col)
    edge_indices, edge_values = ppr.get_ppr(edge_index, alpha=alpha, eps=1e-6)
    edge_indices, edge_values = edge_indices.to(device), edge_values.to(device)
    source_nodes = edge_indices[0]
    unique_sources = torch.unique(source_nodes)
    topk_indices_list = []
    topk_values_list = []
    for src in tqdm(unique_sources, desc='sorting ppr'):
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


def personal_pagerank_fora(edge_index, alpha, topk=100, max_iter: int = 100, device='cuda', num_nodes=None, dataset_name=None, dataset_dir=None, ppr_cache_dir=None, fora_bin=None, fora_work_dir=None, fora_epsilon: float = 0.5, fora_query_batch_size: int = 0, rowptr=None, col=None) -> tuple:
    if num_nodes is None:
        raise ValueError('FORA backend requires num_nodes')
    if dataset_name is None:
        raise ValueError('FORA backend requires dataset_name')

    cache_path = _resolve_cache_path(dataset_dir, dataset_name, ppr_cache_dir, 'fora', topk, alpha)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location='cpu')
        return cached['edge_index'].to(device), cached['edge_values'].to(device)

    fora_bin_path = build_fora_if_needed(fora_bin)
    fora_prefix_dir = _ensure_dir(Path(fora_work_dir).expanduser().resolve() if fora_work_dir is not None else DEFAULT_FORA_WORK_DIR.resolve())
    graph_dir = _ensure_dir(fora_prefix_dir / dataset_name)
    if rowptr is not None and col is not None:
        export_graph_for_fora_from_csr(rowptr=rowptr, col=col, num_nodes=num_nodes, output_dir=graph_dir)
    else:
        if edge_index is None:
            raise ValueError('FORA backend requires edge_index or CSR inputs')
        export_graph_for_fora(edge_index=edge_index, num_nodes=num_nodes, output_dir=graph_dir)

    all_query_nodes = list(range(int(num_nodes)))
    batch_size = int(fora_query_batch_size) if fora_query_batch_size else int(num_nodes)
    batch_size = max(1, batch_size)

    edge_parts = []
    value_parts = []
    for batch_start in range(0, len(all_query_nodes), batch_size):
        query_nodes = all_query_nodes[batch_start: batch_start + batch_size]
        dump_path = graph_dir / f'topk_batch_{batch_start}.tsv'
        _run_fora_topk(
            fora_bin=fora_bin_path,
            prefix_dir=fora_prefix_dir,
            dataset_name=dataset_name,
            alpha=alpha,
            topk=topk,
            epsilon=fora_epsilon,
            query_nodes=query_nodes,
            dump_path=dump_path,
        )
        batch_edge_index, batch_edge_values = _parse_fora_dump(dump_path, device='cpu')
        edge_parts.append(batch_edge_index)
        value_parts.append(batch_edge_values)

    if edge_parts:
        edge_index_out = torch.cat(edge_parts, dim=1)
        edge_values_out = torch.cat(value_parts, dim=0)
    else:
        edge_index_out = torch.empty((2, 0), dtype=torch.long)
        edge_values_out = torch.empty((0,), dtype=torch.float32)

    torch.save({'edge_index': edge_index_out, 'edge_values': edge_values_out}, cache_path)
    return edge_index_out.to(device), edge_values_out.to(device)


def personal_pagerank(edge_index, alpha, topk=100, max_iter: int = 100, device='cuda', backend='torchgeo', num_nodes=None, dataset_name=None, dataset_dir=None, ppr_cache_dir=None, fora_bin=None, fora_work_dir=None, fora_epsilon: float = 0.5, fora_query_batch_size: int = 0, rowptr=None, col=None) -> tuple:
    if backend == 'torchgeo':
        return personal_pagerank_torchgeo(edge_index=edge_index, alpha=alpha, topk=topk, max_iter=max_iter, device=device, rowptr=rowptr, col=col)
    if backend == 'fora':
        return personal_pagerank_fora(
            edge_index=edge_index,
            alpha=alpha,
            topk=topk,
            max_iter=max_iter,
            device=device,
            num_nodes=num_nodes,
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            ppr_cache_dir=ppr_cache_dir,
            fora_bin=fora_bin,
            fora_work_dir=fora_work_dir,
            fora_epsilon=fora_epsilon,
            fora_query_batch_size=fora_query_batch_size,
            rowptr=rowptr,
            col=col,
        )
    raise ValueError(f'Unsupported ppr backend: {backend}')


def metis_partition(csr_adjacency: pymetis.CSRAdjacency, eweights: list[list], n_parts):
    try:
        n_cuts, membership = pymetis.part_graph(nparts=n_parts, adjacency=csr_adjacency, eweights=eweights)
    except Exception as e:
        print(f'Metis failed: {e}')
        raise
    partitions = [[] for _ in range(n_parts)]
    for node_idx, part_id in enumerate(membership):
        partitions[part_id].append(node_idx)
    return [torch.tensor(part, dtype=torch.long) for part in partitions]


def build_adj_fromat(sorted_ppr_matrix):
    print('======start adj format building===========')
    edge_index, ppr_val = sorted_ppr_matrix
    edge_index, ppr_val = edge_index.to('cpu'), ppr_val.to('cpu')
    assert edge_index.shape[0] == 2
    num_nodes = int(edge_index.max().item()) + 1
    src, dst = edge_index[0], edge_index[1]
    u = torch.min(src, dst)
    v = torch.max(src, dst)
    edge_key = u * num_nodes + v
    unique_keys, inverse_indices, _ = torch.unique(edge_key, return_inverse=True, return_counts=True)
    unique_edges = torch.stack([
        torch.div(unique_keys, num_nodes, rounding_mode='floor'),
        unique_keys % num_nodes,
    ], dim=1)
    summed_ppr = torch.zeros(inverse_indices.max() + 1, device=ppr_val.device)
    summed_ppr.scatter_add_(0, inverse_indices, ppr_val)
    weights = (summed_ppr * 1000).clamp_min(1).long().cpu()
    print('======构建无向连接===========')
    u_all = torch.cat([unique_edges[:, 0], unique_edges[:, 1]])
    v_all = torch.cat([unique_edges[:, 1], unique_edges[:, 0]])
    weights_all = torch.cat([weights, weights])
    sort_idx = torch.argsort(u_all)
    u_all = u_all[sort_idx].cpu().numpy()
    v_all = v_all[sort_idx].cpu().numpy()
    weights_all_np = weights_all[sort_idx].numpy()
    print('======csr format building===========')
    xadj = np.zeros(num_nodes + 1, dtype=np.int32)
    degrees = np.bincount(u_all, minlength=num_nodes)
    xadj[1:] = np.cumsum(degrees)
    adjncy = v_all.astype(np.int32)
    eweights = weights_all_np.astype(np.int32)
    assert len(adjncy) == len(eweights)
    assert xadj[-1] == len(adjncy)
    csr_adj = pymetis.CSRAdjacency(adj_starts=xadj.tolist(), adjacent=adjncy.tolist())
    print('======adj weight building===========')
    adj_weight = {}
    unique_u = unique_edges[:, 0].cpu().numpy()
    unique_v = unique_edges[:, 1].cpu().numpy()
    unique_w = weights.numpy()
    for i in tqdm(range(len(unique_u)), desc='adj weight'):
        adj_weight[(int(unique_u[i]), int(unique_v[i]))] = int(unique_w[i])
    return csr_adj, eweights.tolist(), adj_weight


def ppr_partition(sorted_ppr_matrix: list[torch.tensor, torch.tensor], flatten_train_idx, num_set: int):
    train_set = set(flatten_train_idx)
    partitioned_results = []
    print(f'num_of_ppr:{len(sorted_ppr_matrix)}')
    for start_idx in tqdm(range(0, len(sorted_ppr_matrix), num_set), desc='ppr partition'):
        end_idx = min(start_idx + num_set, len(sorted_ppr_matrix))
        node_set = set()
        for j in range(start_idx, end_idx):
            if sorted_ppr_matrix.get(j, None) is None:
                break
            ppr_nodes = [item[0] for item in sorted_ppr_matrix[j]]
            node_set |= set(ppr_nodes)
        if not node_set:
            print('None type found!')
            continue
        partitioned_results.append(list(node_set))
    return [torch.tensor(partition, dtype=torch.long) for partition in partitioned_results]


def add_isolated_connections(ppr_result, num_nodes: int, connect_prob: float = 0.01, ppr_fill_value: float = 0.001, device='cuda') -> tuple:
    edge_index, edge_values = ppr_result
    edge_index = edge_index.to(device)
    edge_values = edge_values.to(device)
    appeared_nodes = torch.unique(edge_index)
    all_nodes = torch.arange(num_nodes, device=device)
    is_isolated = ~torch.isin(all_nodes, appeared_nodes)
    isolated_nodes = all_nodes[is_isolated]
    if isolated_nodes.numel() == 0:
        return edge_index, edge_values
    non_isolated_nodes = all_nodes[~is_isolated]
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


if __name__ == '__main__':
    dataset_dir = './dataset'
    dataset_name = 'cora'
    feature = torch.load(os.path.join(dataset_dir, dataset_name, 'x.pt'))
    edge_index = torch.load(os.path.join(dataset_dir, dataset_name, 'edge_index.pt'))
    num_nodes = feature.shape[0]

    sorted_ppr_matrix = personal_pagerank(edge_index, alpha=0.85, topk=100, num_nodes=num_nodes, dataset_name=dataset_name, dataset_dir=dataset_dir)
    csr_adjacency, eweights, _ = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)
    partitioned_results = metis_partition(csr_adjacency, eweights, 10)
    print(f'idx0:{partitioned_results[0]},num:{len(partitioned_results[0])}')
