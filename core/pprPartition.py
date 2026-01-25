import os
import networkx as nx
import torch
from torch_geometric.utils import ppr
from tqdm import tqdm
import numpy as np
import pymetis

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
        # =====================筛选训练节点全部输入
        # filtered = [n for n in part if n in train_set]
        # if filtered:
        #     filtered_partitions.append(torch.tensor(filtered, dtype=torch.long))
        # =====================不筛选训练节点全部输入
        filtered_partitions.append(torch.tensor(part, dtype=torch.long))
        # 可选：移除 print，避免 I/O 瓶颈
    return filtered_partitions

def build_adj_fromat(sorted_ppr_matrix):
    print("======start adj format building===========")
    edge_index, ppr_val = sorted_ppr_matrix
    edge_index, ppr_val = edge_index.to('cpu'),ppr_val.to('cpu')
    assert edge_index.shape[0] == 2
    num_nodes = int(edge_index.max().item()) + 1
    # BUG: 如果一个图中的最大编号节点是一个孤立点,也就不在 edge_index 中,那么取max自然不是最大编号，反应不了节点数量


    # === Step 1: 规范化边并聚合 PPR 权重 ===
    src, dst = edge_index[0], edge_index[1]
    u = torch.min(src, dst)
    v = torch.max(src, dst)
    # edges = torch.stack([u, v], dim=0).t()  # [E, 2]
    # 合并重复边：使用 unique + scatter_add
    edge_key = u * num_nodes + v  # 唯一键
    _, inverse_indices, counts = torch.unique(
        edge_key, return_inverse=True, return_counts=True
    )
    unique_edges = torch.stack([
        torch.div(_, num_nodes, rounding_mode='floor'),
        _ % num_nodes
    ], dim=1)  # [U, 2]
    summed_ppr = torch.zeros(inverse_indices.max() + 1, device=ppr_val.device)
    summed_ppr.scatter_add_(0, inverse_indices, ppr_val) #若原边中有 (0→1, ppr=0.2) 和 (1→0, ppr=0.3)，则合并后该无向边 (0,1) 的 PPR 总和为 0.5
    weights = (summed_ppr * 1000).clamp_min(1).long().cpu()
    print("======构建无向连接===========")
    # === Step 2: 构建无向邻接（每条边存两次）===
    u_all = torch.cat([unique_edges[:, 0], unique_edges[:, 1]])
    v_all = torch.cat([unique_edges[:, 1], unique_edges[:, 0]])
    weights_all = torch.cat([weights, weights]) # 无向边存了两次，权重也存两次
    # 按源节点排序
    sort_idx = torch.argsort(u_all)
    u_all = u_all[sort_idx].cpu().numpy()
    v_all = v_all[sort_idx].cpu().numpy()
    weights_all_np = weights_all[sort_idx].numpy() # 
    print("======csr format building===========")
    # === Step 3: 构建 CSR ===


    xadj = np.zeros(num_nodes + 1, dtype=np.int32)
    degrees = np.bincount(u_all, minlength=num_nodes)
    xadj[1:] = np.cumsum(degrees)
    adjncy = v_all.astype(np.int32)
    eweights = weights_all_np.astype(np.int32)
    assert len(adjncy) == len(eweights)
    assert xadj[-1] == len(adjncy)
    csr_adj = pymetis.CSRAdjacency(
        adj_starts=xadj.tolist(),
        adjacent=adjncy.tolist()
    )



    # === Step 4: 构建 adj_weight 字典（仅唯一无向边）===
    # 注意：只存 (min, max) -> weight
    print("======adj weight building===========")
    adj_weight = {}
    unique_u = unique_edges[:, 0].cpu().numpy()
    unique_v = unique_edges[:, 1].cpu().numpy()
    unique_w = weights.numpy()
    for i in tqdm(range(len(unique_u)),desc="adj weight"):
        adj_weight[(int(unique_u[i]), int(unique_v[i]))] = int(unique_w[i])
    return csr_adj, eweights.tolist(), adj_weight
    # return adjacency,None
    
def ppr_partition(sorted_ppr_matrix:list[torch.tensor,torch.tensor],flatten_train_idx,num_set:int):
    # 将训练索引转换为集合，用于快速查找
    train_set = set(flatten_train_idx)
    # sorted_ppr_matrix[node] = ((12,ppr_val0),(1,ppr_val1),(22,ppr_val2)...)
    partitioned_results = []
    print(f"num_of_ppr:{len(sorted_ppr_matrix)}")
    for start_idx in tqdm(range(0, len(sorted_ppr_matrix), num_set),desc="ppr partition"):
        end_idx = min(start_idx + num_set, len(sorted_ppr_matrix))
        node_set = set()
        for j in range(start_idx, end_idx):
            if sorted_ppr_matrix.get(j,None) == None:
                break
            ppr_nodes = [item[0] for item in sorted_ppr_matrix[j]]
            node_set |= set(ppr_nodes)
        if not node_set:
            print("None type found!")
            continue
        partitioned_results.append(list(node_set))
    filtered_partitions = []
    for partition in partitioned_results:
        # 过滤：只保留在训练集中的节点
        # =====================筛选训练节点全部输入
        # filtered_nodes = [node for node in partition if node in train_set]
        # # 如果分区还有训练节点，则保留
        # if len(filtered_nodes) > 0:
        #     filtered_partitions.append(torch.tensor(filtered_nodes, dtype=torch.long))
        # print(len(filtered_nodes),end=",")
        # =====================不筛选训练节点全部输入
        filtered_partitions.append(torch.tensor(partition, dtype=torch.long))
    return filtered_partitions

def personal_pagerank(edge_index,alpha,topk=100,max_iter:int=100,device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """为所有节点计算个性化PageRank"""
    edge_indices, edge_values = ppr.get_ppr(
        edge_index, 
        alpha=alpha, 
        eps=1e-6
    )# ppr_matrix shape: (tensor[2,edge_num], edge_num)
    edge_indices,edge_values = edge_indices.to(device),edge_values.to(device)
    source_nodes = edge_indices[0]
    unique_sources = torch.unique(source_nodes)
    topk_indices_list = []
    topk_values_list = []
    for src in tqdm(unique_sources,desc="sorting ppr"):
        # 获取当前源节点的所有边
        mask = source_nodes == src
        src_edges = edge_indices[:, mask]
        src_values = edge_values[mask]
        # 如果当前节点的边数量小于等于 k，全部保留
        if src_values.shape[0] <= topk:
            topk_indices_list.append(src_edges)
            topk_values_list.append(src_values)
        else:
            # 获取 top-k 个最大值的索引
            topk_idx = torch.topk(src_values, topk, largest=True)[1]
            # 保留 top-k 个边
            topk_indices_list.append(src_edges[:, topk_idx])
            topk_values_list.append(src_values[topk_idx])
    # 拼接所有结果
    topk_indices = torch.cat(topk_indices_list, dim=1)
    topk_values = torch.cat(topk_values_list)
    return (topk_indices, topk_values)

if __name__ == "__main__":
    dataset_dir = "./dataset"
    dataset_name = "cora"
    feature = torch.load(os.path.join(dataset_dir, dataset_name, 'x.pt'))
    y = torch.load(os.path.join(dataset_dir, dataset_name, 'y.pt'))
    edge_index = torch.load(os.path.join(dataset_dir, dataset_name, 'edge_index.pt'))
    N = feature.shape[0]

    sorted_ppr_matrix = personal_pagerank(edge_index,alpha=0.85,num_set=100,topk=100)
    csr_adjacency,eweights = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)
    partitioned_results = metis_partition(csr_adjacency,eweights,10)
    print(f"idx0:{partitioned_results[0]},num:{len(partitioned_results[0])}")

    # ======================================== simple test
    # sorted_ppr_matrix = {
    #     0: [(1, 0.3), (2, 0.2)],
    #     1: [(0, 0.3), (2, 0.1), (3, 0.05)],
    #     2: [(0, 0.2), (1, 0.1)],
    #     3: [(1, 0.05)]
    # }
    # print(f"sorted_ppr_matrix:{sorted_ppr_matrix}")

    # csr_adjacency,eweights = build_adj_fromat(sorted_ppr_matrix)
    # print(f"xadj:{csr_adjacency.adj_starts},adjancy:{csr_adjacency.adjacent}")
    # partitions = metis_partition(csr_adjacency,eweights,n_parts=2)
    # print(f"partition :{partitions}")