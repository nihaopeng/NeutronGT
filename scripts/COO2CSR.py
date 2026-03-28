import os
import torch

def edge_index_to_csr(input_path="../dataset//edge_index.pt",
                      output_path="../dataset/edge_index_csr.pt"):
    # 读取 edge_index
    edge_index = torch.load(input_path)

    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"edge_index should be a torch.Tensor, but got {type(edge_index)}")

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index should have shape [2, E], but got {tuple(edge_index.shape)}")

    src = edge_index[0].long()
    dst = edge_index[1].long()

    # 节点数：默认按 edge_index 中最大点编号 + 1
    num_nodes = int(torch.max(edge_index).item()) + 1
    num_edges = src.numel()

    # 按 src 排序，构造 CSR
    perm = torch.argsort(src)
    src_sorted = src[perm]
    dst_sorted = dst[perm]

    # 统计每个节点的出边数
    degree = torch.bincount(src_sorted, minlength=num_nodes)

    # rowptr: 长度为 num_nodes + 1
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.cumsum(degree, dim=0)

    # col: 排序后的目标点
    col = dst_sorted

    # 保存
    csr_data = {
        "rowptr": rowptr,
        "col": col,
    }
    torch.save(csr_data, output_path)

    print(f"Loaded edge_index from: {input_path}")
    print(f"num_nodes = {num_nodes}, num_edges = {num_edges}")
    print(f"Saved CSR to: {output_path}")
    print(f"rowptr shape: {rowptr.shape}, col shape: {col.shape}")


if __name__ == "__main__":
    edge_index_to_csr()