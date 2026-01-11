import os
import networkx as nx
import torch
from tqdm import tqdm
import numpy as np
import pymetis

def metis_partition(csr_adjacency:pymetis.CSRAdjacency,eweights:list[list],flatten_train_idx:np.ndarray,n_parts):
    # 将训练索引转换为集合，用于快速查找
    train_set = set(flatten_train_idx)
    n_cuts, membership = pymetis.part_graph(
        nparts=n_parts,
        adjacency=csr_adjacency,
        eweights=eweights
    ) # !!too large n_parts will cause the error "double free or corruption (!prev)"!!
    # 构建分区列表：每个分区包含该分区的所有节点
    partitions = [[] for _ in range(n_parts)]
    partitioned_results = []
    # 遍历每个节点的分区归属
    for node_idx, partition_id in tqdm(enumerate(membership),desc="building partition"):
        partitions[partition_id].append(node_idx)
    filtered_partitions = []
    for partition in partitions:
        # 过滤：只保留在训练集中的节点
        filtered_nodes = [node for node in partition if node in train_set]
        # 如果分区还有训练节点，则保留
        if len(filtered_nodes) > 0:
            filtered_partitions.append(torch.tensor(filtered_nodes, dtype=torch.long))
        print(len(filtered_nodes),end=",")
    return filtered_partitions

def build_adj_fromat(sorted_ppr_matrix:dict):
    """
    传入的ppr只有所有节点的前topk ppr
    不传入所有的节点是因为完整的ppr构建边权重就相当于每条边权都是2.
    返回pymetis执行分区的数据
    """
    adj_weight = {}
    adjacency = [[] for _ in range(len(sorted_ppr_matrix))] # [[1,2,3],[0,4],[3,4]...]
    # nodes_ppr = ((0,ppr_val),(1,ppr_val1),(2,ppr_val2)...)
    for node,nodes_ppr in tqdm(sorted_ppr_matrix.items(),desc="adj_weight building"):
        for (id,val) in nodes_ppr:
            # edge = (min(node,id),max(node,id))
            edge = (node,id)
            adj_weight[edge] = adj_weight.get(edge, 1) + int(val*1000)
            # adj_weight[edge] = adj_weight.get(edge, 1) + val
            # adj = {(src,dst):weight,...}
            # if edge[1] not in adjacency[edge[0]]:
            #     adjacency[edge[0]].append(edge[1])
            adjacency[edge[0]].append(edge[1])
    xadj = [0]  # 顶点指针数组
    adjncy = []  # 邻接数组
    eweights = []  # 边权重数组
    # 遍历每个顶点
    for i in tqdm(range(len(adjacency)),desc="building adj"):
        neighbors = sorted(adjacency[i])
        adjncy.extend(neighbors)
        xadj.append(xadj[-1] + len(neighbors))
        # 添加对应的边权重到 eweights
        for neighbor in neighbors:
            edge = (min(i, neighbor), max(i, neighbor))
            weight = adj_weight.get(edge, 1)  # 默认权重为1
            # 确保权重是整数且大于0
            if weight <= 0:
                weight = 1
            eweights.append(int(weight))
    assert len(eweights)==len(adjncy)
    csr_adjacency = pymetis.CSRAdjacency(
        adj_starts=xadj,
        adjacent=adjncy
    )
    return csr_adjacency,eweights
    # return adjacency,None
    
def ppr_partition(sorted_ppr_matrix:dict,flatten_train_idx,num_set:int):
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
        filtered_nodes = [node for node in partition if node in train_set]
        # 如果分区还有训练节点，则保留
        if len(filtered_nodes) > 0:
            filtered_partitions.append(torch.tensor(filtered_nodes, dtype=torch.long))
        print(len(filtered_nodes),end=",")
    return partitioned_results

def personal_pagerank(edge_index,alpha,topk=100,max_iter:int=100) -> np.ndarray:
    """为所有节点计算个性化PageRank"""
    G = nx.DiGraph()
    G.add_edges_from(edge_index.T.tolist())
    nodes = list(G.nodes())
    ppr_matrix = {}
    for target_node in tqdm(nodes,desc="running pagerank"):
        personalization = {node: 0 for node in nodes}
        personalization[target_node] = 1
        ppr = nx.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
            max_iter=100,
            tol=1.0e-6
        )# {0:ppr_val,1:ppr_val1,2:ppr_val2...}
        ppr_matrix[target_node] = ppr
    # 查看节点5的个性化PageRank（从节点5出发的随机游走）
    # ppr_matrix = sorted(ppr_matrix.items(), key=lambda x: x[0], reverse=True)
    sorted_ppr_matrix = {}
    for node,nodes_ppr in ppr_matrix.items():
        # nodes_ppr = {0:ppr_val,1:ppr_val1,2:ppr_val2...}
        sorted_ppr_matrix[node] = sorted(nodes_ppr.items(), key=lambda x: x[1], reverse=True)[:topk]
        # sorted_ppr_matrix[node] = ((12,ppr_val0),(1,ppr_val1),(22,ppr_val2)...)
    return sorted_ppr_matrix

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