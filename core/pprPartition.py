import os
import networkx as nx
import torch
from tqdm import tqdm
import numpy as np

def pagerank_partition(edge_index,alpha,num_set,topk=100) -> np.ndarray:
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
        )
        ppr_matrix[target_node] = ppr
    # 查看节点5的个性化PageRank（从节点5出发的随机游走）
    # ppr_matrix = sorted(ppr_matrix.items(), key=lambda x: x[0], reverse=True)
    sorted_ppr_matrix = {}
    for node,nodes_ppr in ppr_matrix.items():
        sorted_ppr_matrix[node] = sorted(nodes_ppr.items(), key=lambda x: x[1], reverse=True)[:topk]
    partitioned_results = []
    for i in tqdm(range(0, len(nodes), num_set),desc="building partition"):
        current_group_nodes = nodes[i:i+num_set]
        node_set = set()
        for node in current_group_nodes:
            # 取节点ID并转为集合
            ppr_nodes = [item[0] for item in sorted_ppr_matrix[node]]
            node_set |= set(ppr_nodes)
        partitioned_results.append(list(node_set))
    return partitioned_results

if __name__ == "__main__":
    dataset_dir = "./dataset"
    dataset_name = "cora"
    feature = torch.load(os.path.join(dataset_dir, dataset_name, 'x.pt'))
    y = torch.load(os.path.join(dataset_dir, dataset_name, 'y.pt'))
    edge_index = torch.load(os.path.join(dataset_dir, dataset_name, 'edge_index.pt'))
    N = feature.shape[0]

    partitioned_results = pagerank_partition(edge_index,alpha=0.85,num_set=100,topk=100)
    print(f"idx0:{partitioned_results[0]},num:{len(partitioned_results[0])}")