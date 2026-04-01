import os

import torch


def load_optional_edge_csr(dataset_dir: str, dataset_name: str):
    dataset_path = os.path.join(dataset_dir, dataset_name)
    edge_csr_path = os.path.join(dataset_path, "edge_index_csr.pt")
    if not os.path.exists(edge_csr_path):
        return None
    edge_csr = torch.load(edge_csr_path, map_location="cpu")
    if not isinstance(edge_csr, dict) or "rowptr" not in edge_csr or "col" not in edge_csr:
        raise KeyError(f"Invalid CSR data in {edge_csr_path}: expected dict with rowptr and col")
    rowptr = edge_csr["rowptr"]
    col = edge_csr["col"]
    if rowptr.dim() != 1 or col.dim() != 1:
        raise ValueError(f"Invalid CSR tensor shapes in {edge_csr_path}: rowptr dim={rowptr.dim()}, col dim={col.dim()}")
    if rowptr.numel() == 0:
        raise ValueError(f"Invalid CSR rowptr in {edge_csr_path}: rowptr must contain at least one element")
    if int(rowptr[0].item()) != 0:
        raise ValueError(f"Invalid CSR rowptr in {edge_csr_path}: rowptr[0] must be 0")
    if int(rowptr[-1].item()) != int(col.numel()):
        raise ValueError(
            f"Invalid CSR tensors in {edge_csr_path}: rowptr[-1]={int(rowptr[-1].item())} does not match col.numel()={int(col.numel())}"
        )
    return {"rowptr": rowptr, "col": col}

def _csr_to_edge_index(edge_csr: dict):
    rowptr = torch.as_tensor(edge_csr["rowptr"], dtype=torch.long, device="cpu")
    col = torch.as_tensor(edge_csr["col"], dtype=torch.long, device="cpu")
    if rowptr.numel() <= 1 or col.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    row_lengths = rowptr[1:] - rowptr[:-1]
    rows = torch.arange(row_lengths.numel(), dtype=torch.long).repeat_interleave(row_lengths)
    return torch.stack([rows, col], dim=0)

def _get_node_degrees_from_csr(edge_csr: dict, num_nodes: int):
    rowptr = torch.as_tensor(edge_csr["rowptr"], dtype=torch.long, device="cpu")
    col = torch.as_tensor(edge_csr["col"], dtype=torch.long, device="cpu")
    out_degree = (rowptr[1:] - rowptr[:-1]).to(torch.long)
    in_degree = torch.bincount(col, minlength=num_nodes).to(torch.long)
    return in_degree + 1, out_degree + 1

def _ensure_edge_index(edge_index, edge_csr_data):
    if edge_index is not None:
        return edge_index
    if edge_csr_data is None:
        raise ValueError("edge_index is required when CSR graph data is unavailable")
    return _csr_to_edge_index(edge_csr_data)
