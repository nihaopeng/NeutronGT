import pymetis
import torch
from tqdm import tqdm
from torch_geometric.utils import subgraph
from core.pprPartition import build_adj_fromat, personal_pagerank
from collections import Counter

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
    def __init__(
            self,
            csr_adjacency:pymetis.CSRAdjacency,
            eweights:list,
            feature:torch.Tensor,
            edge_index:list[torch.Tensor,torch.Tensor],
            n_parts:int,
            related_nodes_topk_rate:int,
            attn_type:str,
            sorted_ppr_matrix:list[torch.Tensor]                   
        ) -> None:
        self.attn_type = attn_type
        self.csr_adjacency = csr_adjacency
        self.eweights = eweights
        self.feature = feature
        self.n_parts = n_parts
        self.global_edge_index = edge_index
        self.partition_num_per_parent = n_parts // 2
        self.parent_partition = self.partition(torch.arange(0,len(csr_adjacency.adj_starts)-1),self.csr_adjacency,self.eweights,2)
        # print(len(torch.arange(0,len(csr_adjacency.adj_starts)-1)))
        # BUG:假设 n 个节点，csr_adjaceny.adj_starts 长度为 num_node + 1，则第一个参数是tensor: [0,1,2,...,num_node]
        # torch.range 已被弃用，改为torch.arange(0,len(csr_adjacency.adj_starts)-1)

        #
        # global_expired_node_buffer：全局节点淘汰池
        # partition_expired_node_num：记录每个分区淘汰了多少节点
        #
        self.global_expired_node_buffer = []
        self.partition_expired_node_num = {}


        #
        # sorted_ppr_matrix
        #
        self.ppr_edge_index, self.ppr_val = sorted_ppr_matrix
        self.ppr_edge_index, self.ppr_val = self.ppr_edge_index.to('cpu'), self.ppr_val.to('cpu')

        self.child_partitions = []
        self.partitioned_results = []
        self.sub_edge_index_for_partition_results = []
        self.dup_indices = None
        self.expanded_edge = [[],[]] # format follow the edge index [2,edge_num]
        # 对父分区进行再次metis
        for parent_id,parent_partition in enumerate(self.parent_partition):
            csr_adjacency,eweight = self._extract_subgraph_csr_eweight(parent_partition)
            self.child_partitions.append(self.partition(parent_partition,csr_adjacency,eweight,self.partition_num_per_parent))
            # self.child_partitions = [[tensor,tensor,...],[tensor,tensor,...]]
            # TODO:将另一个父分区中特征相似的并入。√
            # TODO:将对外有联系的对端节点合并入分区。√
        expanded_child_partitions = []
        for parent_group in self.child_partitions:
            expanded_group = []
            for part in parent_group:
                halo_extended = torch.tensor([])
                # halo: partition包括原本partition的节点和邻居节点
                halo_extended = self._merge_related_nodes(part,related_nodes_topk_rate)
                # feature: partition包括原本的partition的节点和另一个parent分区中和本parent分区中相似度高的node
                feature_extended,expanded_edge_p = self._merge_feature_sim(part.long(), n_nodes=1, current_parent_id=parent_id)
                merged = torch.cat([halo_extended, feature_extended], dim=0)
                merged = torch.unique(merged)  # 自动排序 + 去重
                expanded_group.append(merged)
                self.expanded_edge[0].extend(expanded_edge_p[0])
                self.expanded_edge[1].extend(expanded_edge_p[1])
            expanded_child_partitions.append(expanded_group)
        self.child_partitions = expanded_child_partitions
        self.global_edge_index = torch.cat([self.global_edge_index,torch.tensor(self.expanded_edge)],dim=1)
        # TODO:rerange of partition for kv cache √
        self.dup_nodes_per_partition = self._find_duplicate_nodes_and_rerange()
        self.dup_nodes_per_partition_feature = []
        for parent_group in self.child_partitions:
            for part in parent_group:
                self.partitioned_results.append(part)
                self.sub_edge_index_for_partition_results.append(self._get_sub_edge_index(part))

    def partition(self,partition:torch.Tensor,csr_adjacency:pymetis.CSRAdjacency,eweights:list[list],n_parts:int):
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

        #
        # 子分区的邻接矩阵adjacency的节点编号是对这个分区内的节点从0开始重新编号
        # 实际传入的partition变量是全局的节点id
        # 这里根据partitions[part_id].append(partition[node_idx].item())把分区内的id转换为全局id
        #
        for node_idx, part_id in enumerate(membership):
            partitions[part_id].append(partition[node_idx].item())
        # list to Tensor
        partitions_tensor = []
        for part in partitions:
            partitions_tensor.append(torch.tensor(part, dtype=torch.long))
        return partitions_tensor

    def node_in(self):
        """
        指定partition的id，然后从父分区中补充高权的顶点，<补充的数量需要和剔除的数量相同>
        一定要更新global_edge_index，使用_get_sub_edge_index重新构建分区edge!!!
        """
        for result_partition_global_idx in range(len(self.partitioned_results)):
            # 回补数量
            num_to_recover = self.partition_expired_node_num.get(result_partition_global_idx, 0)
            # 如果不需要补，或者 buffer 是空的，直接返回当前状态
            if num_to_recover <= 0 or not self.global_expired_node_buffer:
                self.partition_expired_node_num[result_partition_global_idx] = 0
                return self.partitioned_results[result_partition_global_idx], \
                    self.sub_edge_index_for_partition_results[result_partition_global_idx]
            current_nodes_global_id = self.partitioned_results[result_partition_global_idx]
            device = 'cpu'
            # 去重，转为 tensor
            unique_candidates = torch.tensor(list(set(self.global_expired_node_buffer)), dtype=torch.long, device=device)
            # 排除掉已经在当前分区里的节点
            is_in_current = torch.isin(unique_candidates, current_nodes_global_id)
            valid_candidates = unique_candidates[~is_in_current]
            if valid_candidates.numel() == 0:
                return self.partitioned_results[result_partition_global_idx], self.sub_edge_index_for_partition_results[result_partition_global_idx]
            # 实际能补的数量
            real_add_num = min(num_to_recover, valid_candidates.size(0))
            # 计算 valid_candidates 中每个节点收到的 PPR 总分
            # Score(candidate) = Sum( PPR(u -> candidate) ) for u in current_global_nodes
            # 找出 PPR 矩阵中，起点是"当前分区节点"的边
            # self.ppr_edge_index[0] 是 Source
            mask_src = torch.isin(self.ppr_edge_index[0], current_nodes_global_id.to(device))
            # 目标节点
            dest = self.ppr_edge_index[1][mask_src]
            edge_ppr_values = self.ppr_val[mask_src]
            # 我们只关心那些"在 Buffer 候选集里"的
            mask_dst = torch.isin(dest, valid_candidates)
            final_dest = dest[mask_dst] # 这些是既被当前分区关注，又在 buffer 里的节点
            final_edge_ppr_values = edge_ppr_values[mask_dst]   # 对应的 PPR 值
            # 聚合分数,把 final_edge_ppr_values 加到 valid_candidates 对应的位置上
            scores = torch.zeros(valid_candidates.size(0), device=device)
            if final_dest.numel() > 0:
                valid_candidates, _ = torch.sort(valid_candidates)
                candidates_indices = torch.searchsorted(valid_candidates, final_dest)
                scores.index_add_(0, candidates_indices, final_edge_ppr_values)
            # 选 PPR 分数最高的 TopK
            _, topk_indices = torch.topk(scores, real_add_num, largest=True)
            selected_new_nodes = valid_candidates[topk_indices]
            # 使用 Counter 进行多重集减法，确保 Buffer 中相同 ID 的数量正确减少
            selected_list = selected_new_nodes.tolist()
            buffer_counter = Counter(self.global_expired_node_buffer)
            selected_counter = Counter(selected_list)
            remaining_counter = buffer_counter - selected_counter
            self.global_expired_node_buffer = list(remaining_counter.elements())
            self.partition_expired_node_num[result_partition_global_idx] = 0
            # 添加节点
            new_global_nodes_combined = torch.cat([current_nodes_global_id, selected_new_nodes])
            new_global_nodes_combined, _ = torch.sort(new_global_nodes_combined)
            self.partitioned_results[result_partition_global_idx] = new_global_nodes_combined
            new_sub_edge_index = self._get_sub_edge_index(new_global_nodes_combined)
            # print(f"partition id:{result_partition_global_idx},before:{len(self.sub_edge_index_for_partition_results[result_partition_global_idx][0])},after:{len(new_sub_edge_index[0])}",end="\r")
            self.sub_edge_index_for_partition_results[result_partition_global_idx] = new_sub_edge_index
            # return new_global_nodes_combined, new_sub_edge_index

    def node_out(self,score_of_partition_list:list[torch.Tensor],remove_ratio:float=0.1):
        """
        根据注意力分数剔除节点，一定要更新global_edge_index，使用_get_sub_edge_index重新构建分区edge!!!
        """
        for result_partition_global_idx in range(len(self.partitioned_results)):
            score_of_partition = score_of_partition_list[result_partition_global_idx]
            device = 'cpu'
            # result_partition_global_idx : 0 ~ number of total children_partition 
            #       idx_parent: which means belongs to which parent_partition (0 or 1)
            #       idx_child : the children id within a parent_partition    
            current_nodes_global_id = self.partitioned_results[result_partition_global_idx].to(device) #[N,]
            current_edges_local_id = self.sub_edge_index_for_partition_results[result_partition_global_idx].to(device) #[2,E]
            current_node_num = current_nodes_global_id.size(0)
            # self.partition_results: list[torch.Tensor],                           [n_part,  partition_node_num]
            # self.sub_edge_index_for_partition_results: list[list[torch.Tensor]],  [n_part,  2,  E]
            # full attention case:  score :[b or 1,num_heads,seq_len,seq_len] 
            # sparse attention case: score :[edge_num,num_head,1]   
            node_scores = torch.zeros(current_node_num, device=device)
            if score_of_partition.dim() == 1:
                assert current_node_num == score_of_partition.shape[0], \
                    f"Node score dim mismatch: nodes={current_node_num}, score={score_of_partition.shape}"
                node_scores = score_of_partition.to(device)
            elif self.attn_type == "full":
                assert self.partitioned_results[result_partition_global_idx].shape[0] == score_of_partition.shape[3], \
                    f"Full Attn dim mismatch: partition_results[{result_partition_global_idx}]={self.partitioned_results[result_partition_global_idx].shape}, score={score_of_partition.shape}"
                assert current_node_num == score_of_partition.shape[3], \
                    f"Full Attn dim mismatch: nodes={current_node_num}, score={score_of_partition.shape}"
                # [1, H, N, N] -> mean heads -> [1, N, N] -> squeeze -> [N, N]
                avg_attn = score_of_partition.mean(dim=1).squeeze(0)
                # [N, N] -> sum -> [N]
                node_scores = avg_attn.sum(dim=0)
            elif self.attn_type == "sparse":
                assert  self.sub_edge_index_for_partition_results[result_partition_global_idx].shape[1] == score_of_partition.shape[0],\
                    f"Sparse Attn dim mismatch: edges={current_edges_local_id.shape[1]}, score={score_of_partition.shape}"
                # [E, H, 1] -> mean heads -> [E]
                edge_scores = score_of_partition.mean(dim=1).squeeze(-1).to(device)
                # 将边权重聚合到 Target 节点 (index 1)
                target_nodes = current_edges_local_id[1]
                assert target_nodes.shape == edge_scores.shape,\
                    f"target_nodes.shape={target_nodes.shape}, edge_scores.shape={edge_scores.shape}"
                node_scores.scatter_add_(0, target_nodes, edge_scores)
            ratio = 1 - remove_ratio # 保留率
            num_keep = int(current_node_num * ratio)
            num_keep = max(num_keep, 1) 
            sorted_indices = torch.argsort(node_scores, descending=True)
            keep_node_indices = sorted_indices[:num_keep]
            expire_node_indices = sorted_indices[num_keep:]
            keep_node_indices, _ = torch.sort(keep_node_indices)
            expire_node_indices, _ = torch.sort(expire_node_indices)
            # add them to expired node buffer
            if expire_node_indices.numel() > 0:
                expired_global_nodes_id = current_nodes_global_id[expire_node_indices].tolist()
                self.partition_expired_node_num[result_partition_global_idx] = len(expired_global_nodes_id)
                self.global_expired_node_buffer.extend(expired_global_nodes_id)
            # new partition nodes global id
            new_partition_global_nodes_id = current_nodes_global_id[keep_node_indices]
            # print(f"partition id:{result_partition_global_idx},before:{len(self.partitioned_results[result_partition_global_idx])},after:{len(new_partition_global_nodes_id)}",end="\r")
            self.partitioned_results[result_partition_global_idx] = new_partition_global_nodes_id
            # new partition edges 
            new_sub_edge_index, _ = subgraph(
                subset=keep_node_indices,
                edge_index=current_edges_local_id, 
                relabel_nodes=True,
                num_nodes=current_node_num
            )
            self.sub_edge_index_for_partition_results[result_partition_global_idx] = new_sub_edge_index
            # return new_partition_global_nodes_id, new_sub_edge_index

    def _get_sub_edge_index(self, node_set: torch.Tensor) -> torch.Tensor:
        sub_edge_index, _ = subgraph(
            node_set,
            self.global_edge_index,
            relabel_nodes=True,
            num_nodes=len(self.csr_adjacency.adj_starts) - 1
        )
        return sub_edge_index


    def _extract_subgraph_csr_eweight(self, parent_nodes: list[int]) -> tuple[pymetis.CSRAdjacency, list[int]]:
        """FROM QWEN
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

    def _find_duplicate_nodes_and_rerange(self):
        """FROM QWEN
        重排各子分区：重复节点置前，返回每个分区开头的重复节点列表
        """
        if not hasattr(self, 'child_partitions') or not self.child_partitions:
            return []
        # 收集所有节点并找出全局重复节点
        all_nodes = [node for group in self.child_partitions for part in group for node in part.tolist()]
        if not all_nodes:
            return []
        unique, counts = torch.unique(torch.tensor(all_nodes), return_counts=True)
        duplicated_set = set(unique[counts >= 2].tolist())
        # 重排每个子分区并收集其开头的重复节点
        dup_nodes_per_partition = []
        new_child_partitions = []
        for parent_group in self.child_partitions:
            new_group = []
            for part in parent_group:
                nodes = part.tolist()
                dup = [n for n in nodes if n in duplicated_set]
                non_dup = [n for n in nodes if n not in duplicated_set]
                reranged = torch.tensor(dup + non_dup, dtype=torch.long)
                new_group.append(reranged)
                if dup:
                    dup_nodes_per_partition.append(torch.tensor(dup, dtype=torch.long))
                else:
                    dup_nodes_per_partition.append(torch.empty(0, dtype=torch.long))
            new_child_partitions.append(new_group)

        self.child_partitions = new_child_partitions
        return dup_nodes_per_partition
    
    def _find_duplicate_edges(self) -> torch.Tensor:
        """FROM QWEN
        
        """
        if not hasattr(self, 'child_partitions') or not self.child_partitions:
            return torch.empty((0,2), dtype=torch.long)
        global_node_num = len(self.csr_adjacency.adj_starts) - 1
        all_global_edges_index_for_partition_results = []
        iterator = zip(self.partitioned_results,self.sub_edge_index_for_partition_results)
        for partition_idx,(global_nodes,local_edge_index) in enumerate(iterator):
            if local_edge_index.numel() == 0:
                continue
            global_src_node = global_nodes[local_edge_index[0]]
            global_dst_node = global_nodes[local_edge_index[1]]
            global_partition_sub_edges_index = torch.stack([global_src_node,global_dst_node],dim = 0)  # [2,partition_edges_num]
            all_global_edges_index_for_partition_results.append(global_partition_sub_edges_index)   
        if not all_global_edges_index_for_partition_results:
            return torch.empty((0, 2), dtype=torch.long)
        total_edges = torch.cat(all_global_edges_index_for_partition_results, dim=1)
        u = total_edges[0].long()
        v = total_edges[1].long()
        edge_keys = u * global_node_num + v
        unique_edge_keys, counts = torch.unique(edge_keys, return_counts=True)
        duplicate_edges_keys = unique_edge_keys[counts >= 2]
        if duplicate_edges_keys.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long)
        dup_u = torch.div(duplicate_edges_keys, global_node_num, rounding_mode='floor')
        dup_v = duplicate_edges_keys % global_node_num
        duplicate_edges_tensor = torch.stack([dup_u, dup_v], dim=1) # [N_dup, 2]
        return duplicate_edges_tensor


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
        for node in tqdm(partition.tolist(),desc="_merge_related_nodes"):
            start, end = xadj[node], xadj[node + 1]   # 节点 u 的邻居存储在 adjncy[xadj[u] : xadj[u+1]]
            deg = end - start
            if deg == 0:
                continue
            neighbors = adjncy[start:end]
            weights = self.eweights[start:end]
            # weights = self.eweights[edge_ptr : edge_ptr + deg]
            # BUG: 为什么要以edge_ptr访问eweight?
            # 已经改为了self.eweights[start:end]
            for nb, w in zip(neighbors, weights):
                if nb not in partition_set:
                    external_neighbors[nb] = external_neighbors.get(nb, 0) + w
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
        centroid = self.feature[partition].mean(dim=0)  # (D,)  本分区的质心
        other_features = self.feature[other_parent_nodes]  # (M, D)  另一个父分区的所有向量
        distances = torch.norm(other_features - centroid, dim=1)  # (M,) 表示other_feature的每个点的特征和质心向量的距离
        num_to_select = min(n_nodes, len(other_parent_nodes))
        _, top_indices = torch.topk(distances, num_to_select, largest=False)# 距离前 num_to_select近的向量的索引
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
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

    
    dataset = "cora"
    try:
        feature = torch.load(f'./dataset/{dataset}/x.pt') 
        edge_index = torch.load(f'./dataset/{dataset}/edge_index.pt') 
        N = feature.shape[0]
        print(f"Loaded {dataset} dataset. Nodes: {N}")
    except FileNotFoundError:
        print("Dataset not found, generating random mock data...")
        N = 1000
        feature = torch.randn(N, 16)
        # 生成随机边
        edge_index = torch.randint(0, N, (2, 5000))
        # 移除自环
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    
    print("Calculating PPR...")
    sorted_ppr_matrix = personal_pagerank(edge_index, 0.85, topk=100)
    csr_adjacency, eweights, adj_weight = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)

    # ------------------ 初始化 Partition ------------------
    n_parts = 4 
    wm = weightMetis_keepParent(
        csr_adjacency=csr_adjacency, 
        eweights=eweights, 
        feature=feature,
        n_parts=n_parts,
        edge_index=edge_index,
        related_nodes_topk_rate=5,
        attn_type="full", # 测试 full attention 模式
        sorted_ppr_matrix=sorted_ppr_matrix
    )

    
    print("\n" + "="*20 + " Initial State " + "="*20)
    for i, res_partition in enumerate(wm.partitioned_results):
        print(f"Partition {i} initial size: {len(res_partition)}")

    
    print("\n" + "="*20 + " Testing node_out " + "="*20)
    fake_attn_score_list = []
    print(f"Executing node_out on Partitions...")
    n_currs = []
    for i in range(n_parts):
        target_part_idx = i  
        current_nodes = wm.partitioned_results[target_part_idx]
        n_curr = len(current_nodes)
        n_currs.append(n_curr)
        # 构造模拟的 Attention Score [1, Heads, N, N]
        num_heads = 4
        fake_attn_score = torch.rand((1, num_heads, n_curr, n_curr), device=feature.device)
        fake_attn_score_list.append(fake_attn_score)
        print(f"  > partition {target_part_idx} Before: {n_curr} nodes")
    print()
    wm.node_out(fake_attn_score_list,remove_ratio=0.1)
    for i in range(n_parts):
        print(f"  > partition {i} After:  {len(wm.partitioned_results[i])} nodes")
        print(f"  > Removed: {n_currs[i] - len(wm.partitioned_results[i])} nodes")
    
        # 验证 Buffer
        print(f"  > Partition {i} has removed {wm.partition_expired_node_num.get(i, 0)} nodes")
    
    print("\n" + "="*20 + " Testing node_in " + "="*20)
    for i in range(n_parts):
        target_part_idx = i 
        current_nodes = wm.partitioned_results[target_part_idx]
        n_curr = len(current_nodes)
        print(f"Executing node_in on Partition {target_part_idx}...")
        print(f"  > partition {target_part_idx} Before Recover: {len(current_nodes)} nodes")
    print()
    wm.node_in()
    for i in range(n_parts):
        print(f"  > After Recover: {len(wm.partitioned_results[i])} nodes")
        
        # 验证
        removed_num = wm.partition_expired_node_num.get(i, 0)
        print(f"  > Partition {i}  still need to recover  {removed_num} nodes (Should be 0)")
