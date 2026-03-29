import torch
from tqdm import tqdm


def build_random_walk_sparse_matrix(edge_index: torch.Tensor, num_nodes: int, device: str):
    edge_index = edge_index.to(device)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    degree = torch.bincount(src, minlength=num_nodes).clamp_min(1)
    values = (1.0 / degree[src]).to(torch.float32)
    indices = torch.stack([dst, src], dim=0)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()


def personal_pagerank_appnp(edge_index, alpha, topk=100, num_iterations: int = 10, batch_size: int = 8, device="cuda") -> tuple:
    edge_index = edge_index.to(device)
    if edge_index.numel() == 0:
        empty_index = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_value = torch.empty((0,), dtype=torch.float32, device=device)
        return empty_index, empty_value

    num_nodes = int(edge_index.max().item()) + 1
    topk = min(int(topk), num_nodes)
    batch_size = max(1, int(batch_size))
    num_iterations = max(1, int(num_iterations))

    transition = build_random_walk_sparse_matrix(edge_index, num_nodes, device)
    topk_indices_batches = []
    topk_values_batches = []

    for start in tqdm(range(0, num_nodes, batch_size), desc="appnp ppr"):
        end = min(start + batch_size, num_nodes)
        seed_nodes = torch.arange(start, end, device=device, dtype=torch.long)
        local_batch = seed_nodes.numel()

        initial = torch.zeros((num_nodes, local_batch), device=device, dtype=torch.float32)
        initial[seed_nodes, torch.arange(local_batch, device=device)] = 1.0
        propagated = initial

        for _ in range(num_iterations):
            propagated = (1.0 - alpha) * torch.sparse.mm(transition, propagated) + alpha * initial

        batch_values, batch_indices = torch.topk(propagated, k=topk, dim=0, largest=True, sorted=True)
        batch_sources = seed_nodes.unsqueeze(0).expand(topk, -1)
        topk_indices_batches.append(torch.stack([batch_sources.reshape(-1), batch_indices.reshape(-1)], dim=0))
        topk_values_batches.append(batch_values.reshape(-1))

        del initial, propagated, batch_values, batch_indices, batch_sources
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(topk_indices_batches, dim=1), torch.cat(topk_values_batches, dim=0)
