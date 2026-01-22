import pymetis
import torch
from tqdm import tqdm

from pprPartition import build_adj_fromat, personal_pagerank


class weightMetis_keepParent:
    """
    带边权的metis划分
    metis划分仍然采用接口调用，不重复造轮子，但是由于我们需要使用到父分区的节点，因此方案如下：
    1，对于划分2n个分区的需求，首先使用pymetis划分为2个分区，保留这2个分区作为父分区
    2，将2个分区分别独立使用metis二划分为2n个分区。需要注意第二步划分不能包括父分区之间的边联系，避免干扰。
    3，迭代存在两个接口，interface1：分区p剔除节点列表s；interface2：分区p新增节点列表s
    4，自动增量计算所有分区间的重复节点。
    5,分区内节点n的邻居如果不在分区内，也将它复制到分区内。

    * 构建：类初始化后自动构建分区、同时自动计算重复顶点。
    * 迭代：训练过程中通过node_out,node_in调整各分区节点，同时自动重计算重复节点。
    * tip：所有的操作的节点都使用他在数据集中的全局索引。
    """
    def __init__(self,csr_adjacency:pymetis.CSRAdjacency,eweights:list[list],feature:torch.Tensor,n_parts:int,related_nodes_topk_rate:int) -> None:
        self.csr_adjacency = csr_adjacency
        self.eweights = eweights
        self.feature = feature
        self.n_parts = n_parts
        self.parent_per_partition_num = n_parts // 2
        self.parent_partition = self.partition(torch.range(0,len(csr_adjacency.adj_starts)),self.csr_adjacency,self.eweights,2)
        self.child_partitions = []
        self.expanded_edge = [[],[]] # format follow the edge index [2,edge_num]
        for parent_id,parent_partition in enumerate(self.parent_partition):
            csr_adjacency,eweight = self._extract_subgraph_csr_eweight(parent_partition)
            # 从两个父分区再进行分区。
            self.child_partitions.append(self.partition(parent_partition,csr_adjacency,eweight,self.parent_per_partition_num,is_child=True))
            # self.child_partition = [[tensor,tensor,...],[tensor,tensor,...]]
            # TODO:将另一个父分区中特征相似的并入。
            # TODO:将对外有联系的对端节点合并入分区。
            expanded_child_partitions = []
            for parent_group in self.child_partitions:
                expanded_group = []
                for part in parent_group:
                    # 1. 基于原始分区做 halo 扩展
                    halo_extended = torch.tensor([])
                    halo_extended = self._merge_related_nodes(part,related_nodes_topk_rate)
                    # 2. 基于原始分区做特征相似性扩展
                    # print(part.long())
                    feature_extended,expanded_edge_p = self._merge_feature_sim(part.long(), n_nodes=1, current_parent_id=parent_id)
                    # 3. 合并两者并去重
                    merged = torch.cat([halo_extended, feature_extended], dim=0)
                    merged = torch.unique(merged)  # 自动排序 + 去重
                    expanded_group.append(merged)
                    self.expanded_edge[0].extend(expanded_edge_p[0])
                    self.expanded_edge[1].extend(expanded_edge_p[1])
                expanded_child_partitions.append(expanded_group)
            self.child_partitions = expanded_child_partitions
        self.duplicated_nodes = self._find_duplicate_nodes()

    def partition(self,partition:torch.Tensor,csr_adjacency:pymetis.CSRAdjacency,eweights:list[list],n_parts:int,is_child:bool=False):
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
            partitions[part_id].append(partition[node_idx].item())
        filtered_partitions = []
        for part in partitions:
            filtered_partitions.append(torch.tensor(part, dtype=torch.long))
        return filtered_partitions
    
    def _extract_subgraph_csr_eweight(self, parent_nodes: list[int]) -> tuple[pymetis.CSRAdjacency, list[int]]:
        """
        从原始图中提取由parent_nodes构成的子图，并返回CSR邻接结构和对应的边权列表。
        """
        if len(parent_nodes) == 0:
            return pymetis.CSRAdjacency(xadj=[0], adjncy=[]), []
        if len(parent_nodes) == 1:
            return pymetis.CSRAdjacency(xadj=[0, 0], adjncy=[]), []
        xadj, adjncy = self.csr_adjacency.adj_starts, self.csr_adjacency.adjacent
        eweights = self.eweights
        parent_set = set(parent_nodes)
        node_old_to_new = {old_id: new_id for new_id, old_id in enumerate(parent_nodes)}
        sub_xadj = [0]
        sub_adjncy = []
        sub_eweights = []
        # 用于遍历 eweights：METIS 要求边权顺序与 adjncy 完全一致
        edge_idx = 0
        for u in tqdm(parent_nodes,desc="_extract_subgraph_csr_eweight"):
            start, end = xadj[u], xadj[u + 1]
            local_neighbors = []
            local_weights = []
            for i in range(start, end):
                v = adjncy[i]
                w = eweights[edge_idx]
                edge_idx += 1
                if v in parent_set:
                    local_neighbors.append(v)
                    local_weights.append(w)
            # 映射到新节点 ID
            sub_adjncy.extend(node_old_to_new[v] for v in local_neighbors)
            sub_eweights.extend(local_weights)
            sub_xadj.append(len(sub_adjncy))
        return pymetis.CSRAdjacency(adj_starts=sub_xadj, adjacent=sub_adjncy), sub_eweights

    def _find_duplicate_nodes(self) -> torch.Tensor:
        """
        找出重复节点。
        """
        if not hasattr(self, 'child_partitions') or not self.child_partitions:
            return torch.empty(0, dtype=torch.long)
        all_tensors = []
        for parent_group in self.child_partitions:
            all_tensors.extend(parent_group)
        if not all_tensors:
            return torch.empty(0, dtype=torch.long)
        # 拼接所有节点
        all_nodes = torch.cat(all_tensors, dim=0)
        # 统计重复
        unique_nodes, counts = torch.unique(all_nodes, return_counts=True)
        duplicate_nodes = unique_nodes[counts >= 2]
        return duplicate_nodes
    
    def _merge_related_nodes(self, partition: torch.Tensor, topk_percent:int) -> torch.Tensor:
        """
        将分区 partition 中所有节点的外部邻居（即不在 partition 中的邻居）合并进来，
        返回扩展后的分区（全局节点索引）。
        """
        print("==正在合并关联节点==")
        xadj = self.csr_adjacency.adj_starts
        adjncy = self.csr_adjacency.adjacent
        partition_set = set(partition.tolist())
        external_neighbors = {}
        edge_ptr = 0  # 全局边指针
        for node in tqdm(partition.tolist(),desc="_merge_related_nodes"):
            start, end = xadj[node], xadj[node + 1]
            deg = end - start
            if deg == 0:
                continue
            neighbors = adjncy[start:end]
            weights = self.eweights[edge_ptr : edge_ptr + deg]
            for nb, w in zip(neighbors, weights):
                if nb not in partition_set:
                    external_neighbors[nb] = external_neighbors.get(nb, 0) + w
            edge_ptr += deg
        # 排序并选 topk%
        sorted_items = sorted(external_neighbors.items(), key=lambda x: x[1], reverse=True)
        n_select = len(sorted_items) * topk_percent // 100 if topk_percent > 0 else 0
        selected_nodes = {node for node, _ in sorted_items[:n_select]}
        expanded_nodes = list(partition_set | selected_nodes)
        expanded_nodes.sort()
        return torch.tensor(expanded_nodes, dtype=torch.long)
    
    def _merge_feature_sim(
        self,
        partition: torch.Tensor,
        n_nodes: int,
        current_parent_id: int,
        connect_prob: float = 0.01
    ) -> tuple[torch.Tensor, torch.Tensor]:
        print("==正在合并特征相似节点==")
        assert current_parent_id in (0, 1), "Only two parent partitions supported."
        other_parent_id = 1 - current_parent_id
        other_parent_nodes = self.parent_partition[other_parent_id]
        if partition.numel() == 0:
            return partition, torch.empty((2, 0), dtype=torch.long)
        # Step 1: 选最相似的 n_nodes 个节点
        centroid = self.feature[partition].mean(dim=0)  # (D,)
        other_features = self.feature[other_parent_nodes]  # (M, D)
        distances = torch.norm(other_features - centroid, dim=1)  # (M,)
        num_to_select = min(n_nodes, len(other_parent_nodes))
        _, top_indices = torch.topk(distances, num_to_select, largest=False)
        selected_nodes = other_parent_nodes[top_indices]  # 全局索引
        # 合并节点
        merged = torch.unique(torch.cat([partition, selected_nodes], dim=0))
        # Step 2: 生成虚拟边（仅新边，无向）
        virtual_edges = []
        for new_node in selected_nodes.tolist():
            for old_node in partition.tolist():
                if torch.rand(1).item() < connect_prob:
                    # 无向边：双向
                    virtual_edges.append([old_node, new_node])
                    virtual_edges.append([new_node, old_node])
        # 转为 edge_index
        if virtual_edges:
            virtual_edge_index = torch.tensor(virtual_edges, dtype=torch.long).t()  # [2, E]
        else:
            virtual_edge_index = torch.empty((2, 0), dtype=torch.long)
        return merged, virtual_edge_index

if __name__ == "__main__":
    # ---------------------cora test------------------------
    # dataset = "ogbn-arxiv"
    dataset = "pubmed"
    feature = torch.load(f'./dataset/{dataset}/x.pt') # [N, x_dim]
    y = torch.load(f'./dataset/{dataset}/y.pt') # [N]
    edge_index = torch.load(f'./dataset/{dataset}/edge_index.pt') # [2, num_edges]
    N = feature.shape[0]
    # ---------------------cora test------------------------
    # # ---------------------mini test------------------------
    # edge_index = torch.tensor([
    #     [0, 1, 2, 1, 2, 0, 3, 4, 5, 4, 5, 3, 2, 3],
    #     [1, 2, 0, 0, 1, 2, 4, 5, 3, 3, 4, 5, 3, 2]
    # ], dtype=torch.long)
    # # 特征：让 A 组（0,1,2）特征接近 [1,0]，B 组（3,4,5）接近 [0,1]
    # feature = torch.tensor([[1.0, 0.0],[1.0, 0.1],[0.9, -0.1],[0.0, 1.0],[0.1, 1.0],[-0.1, 0.9]])
    # N = feature.shape[0]
    # # ---------------------mini test------------------------
    sorted_ppr_matrix = personal_pagerank(edge_index,0.85,topk=100)
    csr_adjacency,eweights,adj_weight = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)
    # ------------------ 初始化 weightMetis_keepParent ------------------
    n_parts = 10  # 划分为4个子分区
    wm = weightMetis_keepParent(csr_adjacency=csr_adjacency, eweights=eweights, n_parts=n_parts,feature=feature,related_nodes_topk_rate=5)
    # ------------------ 输出结果 ------------------
    # 构造 CSRAdjacency 对象
    print("node_num:",len(csr_adjacency.adj_starts)-1)
    print("xadj num:", len(csr_adjacency.adj_starts))
    print("adjncy num:", len(csr_adjacency.adjacent))
    print("eweights num:", len(eweights))
    print("\n=== 父分区 ===")
    for i, part in enumerate(wm.parent_partition):
        print(f"Parent {i} num: {len(part.tolist())}")
    print("\n=== 子分区 ===")
    for i, child_parts in enumerate(wm.child_partitions):
        print(f"From Parent {i}:")
        for j, part in enumerate(child_parts):
            print(f"  Child {j} num: {len(part.tolist())}")
    print("\n=== 重复节点（全局ID） ===")
    print("Duplicate nodes num:", len(wm.duplicated_nodes.tolist()))
    print("\n=== 虚拟边数量 ===")
    print("virtual edge num:", len(wm.expanded_edge[0]))