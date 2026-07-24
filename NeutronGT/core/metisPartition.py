import pymetis
import time
import torch
from tqdm import tqdm
from torch_geometric.utils import subgraph
from core.ppr_preprocess import build_adj_fromat, personal_pagerank
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
            edge_index:list[torch.Tensor,torch.Tensor],
            edge_csr_data:dict | None,
            n_parts:int,
            attn_type:str,
            sorted_ppr_matrix:list[torch.Tensor],
            window_aug_strategy:str='ours',
            window_extra_node_ratio:float=0.30,
            window_related_ratio:float=0.15,
            window_hub_ratio:float=0.15,
            seed:int=42,
        ) -> None:
        self.attn_type = attn_type
        self.csr_adjacency = csr_adjacency
        self.eweights = eweights
        self.n_parts = n_parts
        self.global_edge_index = edge_index
        self.num_nodes = len(self.csr_adjacency.adj_starts) - 1 if hasattr(self.csr_adjacency, 'adj_starts') else None
        self.original_rowptr, self.original_col = self._prepare_original_graph_csr(edge_index, edge_csr_data)
        self.partition_num_per_parent = n_parts // 2
        self.window_aug_strategy = window_aug_strategy
        self.window_extra_node_ratio = float(window_extra_node_ratio)
        self.window_related_ratio = float(window_related_ratio)
        self.window_hub_ratio = float(window_hub_ratio)
        self.seed = int(seed)
        self.hub_node_order = None
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
        # 细分计时保留在 timing_stats 中，便于上层组合 Stage 1 / Stage 2。
        self.timing_stats = {
            'parent_partition_time': 0.0,
            'child_partition_time': 0.0,
            'related_nodes_merge_time': 0.0,
            'hub_node_merge_time': 0.0,
            'random_fill_time': 0.0,
            'augmentation_target_extra_nodes': 0,
            'augmentation_related_nodes': 0,
            'augmentation_hub_nodes': 0,
            'augmentation_filler_nodes': 0,
            'expanded_edge_concat_time': 0.0,
            'duplicate_rerange_time': 0.0,
            'subgraph_build_time': 0.0,
        }
        partition_build_start = time.time()
        parent_partition_start = time.time()
        self.parent_partition = self.partition(torch.arange(0,len(csr_adjacency.adj_starts)-1),self.csr_adjacency,self.eweights,2)
        self.timing_stats['parent_partition_time'] = time.time() - parent_partition_start
        child_partition_start = time.time()
        # 对父分区进行再次metis
        for parent_id,parent_partition in enumerate(self.parent_partition):
            csr_adjacency,eweight = self._extract_subgraph_csr_eweight(parent_partition)
            self.child_partitions.append(self.partition(parent_partition,csr_adjacency,eweight,self.partition_num_per_parent))
            # self.child_partitions = [[tensor,tensor,...],[tensor,tensor,...]]
            # TOD:将对外有联系的对端节点合并入分区。√
        self.timing_stats['child_partition_time'] = time.time() - child_partition_start
        if self.window_aug_strategy in ('hub', 'ours'):
            hub_start = time.time()
            self.hub_node_order = self._compute_hub_node_order()
            self.timing_stats['hub_node_merge_time'] += time.time() - hub_start

        expanded_child_partitions = []
        for parent_id, parent_group in enumerate(self.child_partitions):
            expanded_group = []
            for child_idx, part in enumerate(parent_group):
                part = part.long()
                partition_id = parent_id * self.partition_num_per_parent + child_idx
                merged, expanded_edge_p = self._augment_partition_equal_size(
                    core_partition=part,
                    parent_id=parent_id,
                    child_idx=child_idx,
                    partition_id=partition_id,
                )
                expanded_group.append(merged)
                self.expanded_edge[0].extend(expanded_edge_p[0].tolist())
                self.expanded_edge[1].extend(expanded_edge_p[1].tolist())
            expanded_child_partitions.append(expanded_group)
        self.child_partitions = expanded_child_partitions
        expanded_edge_concat_start = time.time()
        if self.expanded_edge[0]:
            expanded_edge_tensor = torch.tensor(
                self.expanded_edge,
                dtype=self.global_edge_index.dtype,
            )
            self.global_edge_index = torch.cat([self.global_edge_index, expanded_edge_tensor], dim=1)
        self.timing_stats['expanded_edge_concat_time'] = time.time() - expanded_edge_concat_start
        # TODO:rerange of partition for kv cache √
        duplicate_start = time.time()
        self.dup_nodes_per_partition = self._find_duplicate_nodes_and_rerange()
        self.timing_stats['duplicate_rerange_time'] = time.time() - duplicate_start
        self.dup_nodes_per_partition_feature = []
        # 先构建 partitioned_results
        for parent_group in self.child_partitions:
            for part in parent_group:
                self.partitioned_results.append(part)
        # 一次性扫描 global_edge_index 为所有分区构建子图边索引
        subgraph_build_start = time.time()
        self.sub_edge_index_for_partition_results = self._build_all_sub_edge_indices()
        self.timing_stats['subgraph_build_time'] = time.time() - subgraph_build_start

        # 释放预处理中间数据：Metis 划分完成后不再需要
        del self.csr_adjacency, self.eweights
        del self.ppr_edge_index, self.ppr_val

    def _prepare_original_graph_csr(self, edge_index: torch.Tensor, edge_csr_data: dict | None):
        if edge_csr_data is not None:
            return (
                torch.as_tensor(edge_csr_data["rowptr"], device='cpu'),
                torch.as_tensor(edge_csr_data["col"], device='cpu'),
            )

        num_nodes = int(self.num_nodes) if self.num_nodes is not None else 0
        if num_nodes <= 0 and edge_index is not None and edge_index.numel() > 0:
            num_nodes = int(edge_index.max().item()) + 1
            self.num_nodes = num_nodes
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros((num_nodes + 1,), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        src = edge_index[0].to(device='cpu', dtype=torch.long)
        dst = edge_index[1].to(device='cpu', dtype=torch.long)
        order = torch.argsort(src)
        src = src[order]
        dst = dst[order]
        counts = torch.bincount(src, minlength=num_nodes)
        rowptr = torch.zeros((num_nodes + 1,), dtype=torch.long)
        rowptr[1:] = torch.cumsum(counts, dim=0)
        return rowptr, dst

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
        """单个分区的子图边索引提取（保留兼容 node_in/node_out）"""
        sub_edge_index, _ = subgraph(
            node_set,
            self.global_edge_index,
            relabel_nodes=True,
            num_nodes=self.num_nodes
        )
        return sub_edge_index

    def _build_all_sub_edge_indices(self) -> list:
        """一次性扫描 global_edge_index，为所有分区构建子图边索引。

        替代原有的逐分区 torch_geometric.utils.subgraph() 调用，
        将 50 次全量边扫描合并为 1 次。
        """
        return self._build_all_sub_edge_indices_multi_membership()

    def _build_all_sub_edge_indices_multi_membership(self) -> list:
        """Build subgraph edges when copied nodes may belong to many windows."""
        num_parts = len(self.partitioned_results)
        if num_parts == 0:
            return []

        num_nodes = self.num_nodes or int(self.global_edge_index.max().item()) + 1
        node_to_part = torch.full((num_nodes,), -1, dtype=torch.int32)
        node_to_local = torch.full((num_nodes,), -1, dtype=torch.int32)
        duplicate_memberships: dict[int, list[tuple[int, int]]] = {}

        for pid, part in enumerate(self.partitioned_results):
            for local_idx, node in enumerate(part.to(torch.long).cpu().tolist()):
                node = int(node)
                previous_pid = int(node_to_part[node].item())
                if previous_pid == -1:
                    node_to_part[node] = pid
                    node_to_local[node] = local_idx
                elif previous_pid >= 0:
                    previous_local = int(node_to_local[node].item())
                    duplicate_memberships[node] = [(previous_pid, previous_local), (pid, local_idx)]
                    node_to_part[node] = -2
                    node_to_local[node] = -1
                else:
                    duplicate_memberships[node].append((pid, local_idx))

        src_all = self.global_edge_index[0]
        dst_all = self.global_edge_index[1]
        total_edges = src_all.numel()
        CHUNK = 50_000_000

        part_srcs: list[list] = [[] for _ in range(num_parts)]
        part_dsts: list[list] = [[] for _ in range(num_parts)]
        part_src_extra: list[list[int]] = [[] for _ in range(num_parts)]
        part_dst_extra: list[list[int]] = [[] for _ in range(num_parts)]

        for chunk_start in range(0, total_edges, CHUNK):
            chunk_end = min(chunk_start + CHUNK, total_edges)
            src = src_all[chunk_start:chunk_end]
            dst = dst_all[chunk_start:chunk_end]

            src_part = node_to_part[src]
            dst_part = node_to_part[dst]
            src_local = node_to_local[src]
            dst_local = node_to_local[dst]

            single_same_part = (src_part >= 0) & (src_part == dst_part)
            if single_same_part.any():
                valid_part = src_part[single_same_part]
                valid_src_local = src_local[single_same_part]
                valid_dst_local = dst_local[single_same_part]
                order = torch.argsort(valid_part)
                valid_part = valid_part[order]
                valid_src_local = valid_src_local[order]
                valid_dst_local = valid_dst_local[order]
                boundaries = torch.searchsorted(
                    valid_part,
                    torch.arange(num_parts + 1, device=valid_part.device)
                )
                for pid in range(num_parts):
                    s, e = int(boundaries[pid].item()), int(boundaries[pid + 1].item())
                    if s == e:
                        continue
                    part_srcs[pid].append(valid_src_local[s:e].cpu())
                    part_dsts[pid].append(valid_dst_local[s:e].cpu())
                del valid_part, valid_src_local, valid_dst_local, order, boundaries

            has_duplicate_endpoint = (src_part == -2) | (dst_part == -2)
            if has_duplicate_endpoint.any():
                dup_src = src[has_duplicate_endpoint].to(torch.long).cpu().tolist()
                dup_dst = dst[has_duplicate_endpoint].to(torch.long).cpu().tolist()
                dup_src_part = src_part[has_duplicate_endpoint].cpu().tolist()
                dup_dst_part = dst_part[has_duplicate_endpoint].cpu().tolist()
                dup_src_local = src_local[has_duplicate_endpoint].cpu().tolist()
                dup_dst_local = dst_local[has_duplicate_endpoint].cpu().tolist()

                for u, v, sp, dp, sl, dl in zip(
                    dup_src,
                    dup_dst,
                    dup_src_part,
                    dup_dst_part,
                    dup_src_local,
                    dup_dst_local,
                ):
                    if sp == -1 or dp == -1:
                        continue
                    src_members = duplicate_memberships.get(int(u)) if sp == -2 else [(int(sp), int(sl))]
                    dst_members = duplicate_memberships.get(int(v)) if dp == -2 else [(int(dp), int(dl))]
                    if not src_members or not dst_members:
                        continue
                    dst_by_pid = {pid: local for pid, local in dst_members}
                    for pid, src_local_idx in src_members:
                        dst_local_idx = dst_by_pid.get(pid)
                        if dst_local_idx is None:
                            continue
                        part_src_extra[pid].append(src_local_idx)
                        part_dst_extra[pid].append(dst_local_idx)

            del src, dst, src_part, dst_part, src_local, dst_local, single_same_part, has_duplicate_endpoint

        sub_edge_list = []
        for pid in range(num_parts):
            if part_src_extra[pid]:
                part_srcs[pid].append(torch.tensor(part_src_extra[pid], dtype=torch.int32))
                part_dsts[pid].append(torch.tensor(part_dst_extra[pid], dtype=torch.int32))
            if not part_srcs[pid]:
                sub_edge_list.append(torch.empty((2, 0), dtype=torch.long))
                continue
            cat_src = torch.cat(part_srcs[pid]).to(torch.long)
            cat_dst = torch.cat(part_dsts[pid]).to(torch.long)
            sub_edge_list.append(torch.stack([cat_src, cat_dst], dim=0))
            del cat_src, cat_dst
            part_srcs[pid] = None
            part_dsts[pid] = None

        del node_to_part, node_to_local, duplicate_memberships
        return sub_edge_list


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
        for u in parent_nodes:
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
        # 收集所有节点并找出全局重复节点（使用 tensor 操作避免 Python list 内存爆炸）
        parts_list = [part for group in self.child_partitions for part in group]
        if not parts_list:
            return []
        all_nodes = torch.cat(parts_list)
        unique, counts = torch.unique(all_nodes, return_counts=True)
        # 重排每个子分区并收集其开头的重复节点
        dup_nodes_per_partition = []
        new_child_partitions = []
        for parent_group in self.child_partitions:
            new_group = []
            for part in parent_group:
                # 使用 tensor isin 替代 Python list 遍历
                is_dup = torch.isin(part, unique[counts >= 2])
                dup = part[is_dup]
                non_dup = part[~is_dup]
                reranged = torch.cat([dup, non_dup], dim=0)
                new_group.append(reranged)
                dup_nodes_per_partition.append(dup if dup.numel() > 0 else torch.empty(0, dtype=torch.long))
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

    def _target_extra_count(self, core_partition: torch.Tensor) -> int:
        if core_partition.numel() == 0 or self.num_nodes is None:
            return 0
        raw_target = int(core_partition.numel() * max(self.window_extra_node_ratio, 0.0))
        max_possible = max(int(self.num_nodes) - int(torch.unique(core_partition).numel()), 0)
        return min(raw_target, max_possible)

    def _compute_hub_node_order(self) -> torch.Tensor:
        if self.num_nodes is None or self.num_nodes <= 0:
            return torch.empty((0,), dtype=torch.long)
        rowptr = self.original_rowptr.to(device='cpu')
        col = self.original_col.to(device='cpu')
        out_degree = (rowptr[1:].to(torch.long) - rowptr[:-1].to(torch.long)).clamp_min(0)
        degree_score = out_degree
        chunk_size = 100_000_000
        in_degree = torch.zeros(int(self.num_nodes), dtype=torch.long)
        for start in range(0, int(col.numel()), chunk_size):
            end = min(start + chunk_size, int(col.numel()))
            in_degree += torch.bincount(col[start:end].to(torch.long), minlength=int(self.num_nodes))
        degree_score += in_degree
        hub_order = torch.argsort(degree_score, descending=True)
        del rowptr, col, out_degree, in_degree, degree_score
        return hub_order.to(torch.long).cpu()

    def _select_from_ordered_candidates(
        self,
        candidates: torch.Tensor,
        selected_set: set[int],
        max_nodes: int,
    ) -> list[int]:
        if max_nodes <= 0 or candidates.numel() == 0:
            return []
        selected = []
        for node in candidates.to(torch.long).cpu().tolist():
            node = int(node)
            if node in selected_set:
                continue
            selected.append(node)
            selected_set.add(node)
            if len(selected) >= max_nodes:
                break
        return selected

    def _select_shared_filler_nodes(
        self,
        selected_set: set[int],
        max_nodes: int,
        partition_id: int,
    ) -> list[int]:
        if max_nodes <= 0 or self.num_nodes is None or self.num_nodes <= 0:
            return []
        max_nodes = min(max_nodes, max(self.num_nodes - len(selected_set), 0))
        if max_nodes <= 0:
            return []

        selected = []
        gen = torch.Generator(device='cpu')
        gen.manual_seed((self.seed + 1_000_003 * (partition_id + 1) + 17) % (2**63 - 1))
        attempts = 0
        while len(selected) < max_nodes and attempts < 20:
            need = max_nodes - len(selected)
            batch_size = min(max(need * 4, 1024), 1_000_000)
            samples = torch.randint(0, self.num_nodes, (batch_size,), generator=gen, dtype=torch.long)
            for node in samples.tolist():
                node = int(node)
                if node in selected_set:
                    continue
                selected.append(node)
                selected_set.add(node)
                if len(selected) >= max_nodes:
                    break
            attempts += 1

        if len(selected) < max_nodes:
            start = (self.seed + 97_531 * (partition_id + 1) + 17) % self.num_nodes
            for offset in range(self.num_nodes):
                node = int((start + offset) % self.num_nodes)
                if node in selected_set:
                    continue
                selected.append(node)
                selected_set.add(node)
                if len(selected) >= max_nodes:
                    break
        return selected

    def _select_related_nodes(
        self,
        partition: torch.Tensor,
        max_nodes: int | None = None,
    ) -> torch.Tensor:
        xadj = self.original_rowptr
        adjncy = self.original_col
        partition_set = set(int(x) for x in partition.tolist())
        external_neighbors = {}
        for node in partition.tolist():
            node = int(node)
            start, end = int(xadj[node].item()), int(xadj[node + 1].item())
            if end <= start:
                continue
            neighbors = adjncy[start:end]
            for nb in neighbors.tolist():
                nb = int(nb)
                if nb not in partition_set:
                    external_neighbors[nb] = external_neighbors.get(nb, 0) + 1

        sorted_items = sorted(external_neighbors.items(), key=lambda x: x[1], reverse=True)
        if max_nodes is not None:
            n_select = min(len(sorted_items), max_nodes)
        else:
            n_select = len(sorted_items)
        selected_nodes = [node for node, _ in sorted_items[:n_select]]
        return torch.tensor(selected_nodes, dtype=torch.long)

    def _merge_related_nodes(self, partition: torch.Tensor) -> torch.Tensor:
        """
        将分区 partition 中所有节点的外部邻居（即不在 partition 中的邻居）合并进来，
        返回扩展后的分区（全局节点索引）。
        """
        selected_nodes = self._select_related_nodes(partition)
        return torch.unique(torch.cat([partition.to(torch.long), selected_nodes], dim=0))

    def _augment_partition_equal_size(
        self,
        core_partition: torch.Tensor,
        parent_id: int,
        child_idx: int,
        partition_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        core_partition = torch.unique(core_partition.to(torch.long))
        target_extra = self._target_extra_count(core_partition)
        self.timing_stats['augmentation_target_extra_nodes'] += int(target_extra)
        selected_set = set(int(x) for x in core_partition.tolist())
        selected_by_source = {'related': [], 'hub': []}
        filler_nodes = []

        def remaining() -> int:
            return target_extra - (len(selected_set) - int(core_partition.numel()))

        def append_candidates(source: str, candidates: torch.Tensor, max_nodes: int):
            take = min(max_nodes, remaining())
            chosen = self._select_from_ordered_candidates(candidates, selected_set, take)
            if source in selected_by_source:
                selected_by_source[source].extend(chosen)

        if target_extra <= 0:
            return core_partition, torch.empty((2, 0), dtype=torch.long)

        if self.window_aug_strategy == 'random':
            random_start = time.time()
            filler_nodes.extend(self._select_shared_filler_nodes(selected_set, remaining(), partition_id))
            self.timing_stats['random_fill_time'] += time.time() - random_start
        elif self.window_aug_strategy == 'hub':
            if self.hub_node_order is None:
                hub_start = time.time()
                self.hub_node_order = self._compute_hub_node_order()
                self.timing_stats['hub_node_merge_time'] += time.time() - hub_start
            hub_start = time.time()
            append_candidates('hub', self.hub_node_order, remaining())
            self.timing_stats['hub_node_merge_time'] += time.time() - hub_start
        elif self.window_aug_strategy == 'related':
            related_start = time.time()
            related_candidates = self._select_related_nodes(core_partition, max_nodes=target_extra)
            append_candidates('related', related_candidates, remaining())
            self.timing_stats['related_nodes_merge_time'] += time.time() - related_start
        elif self.window_aug_strategy == 'ours':
            related_quota = min(int(core_partition.numel() * max(self.window_related_ratio, 0.0)), target_extra)
            hub_quota = min(int(core_partition.numel() * max(self.window_hub_ratio, 0.0)), target_extra)

            related_candidates = torch.empty(0, dtype=torch.long)
            if related_quota > 0:
                related_start = time.time()
                related_candidates = self._select_related_nodes(core_partition, max_nodes=target_extra)
                append_candidates('related', related_candidates, related_quota)
                self.timing_stats['related_nodes_merge_time'] += time.time() - related_start

            if hub_quota > 0:
                if self.hub_node_order is None:
                    hub_start = time.time()
                    self.hub_node_order = self._compute_hub_node_order()
                    self.timing_stats['hub_node_merge_time'] += time.time() - hub_start
                hub_start = time.time()
                append_candidates('hub', self.hub_node_order, hub_quota)
                self.timing_stats['hub_node_merge_time'] += time.time() - hub_start

            # If one source lacks enough unique candidates, keep source priority before random fallback.
            if related_quota > 0:
                append_candidates('related', related_candidates, remaining())
            if hub_quota > 0:
                append_candidates('hub', self.hub_node_order, remaining())
        else:
            raise ValueError(f'Unsupported window_aug_strategy: {self.window_aug_strategy}')

        if remaining() > 0:
            random_start = time.time()
            filler_nodes.extend(self._select_shared_filler_nodes(selected_set, remaining(), partition_id))
            self.timing_stats['random_fill_time'] += time.time() - random_start
        self.timing_stats['augmentation_related_nodes'] += len(selected_by_source['related'])
        self.timing_stats['augmentation_hub_nodes'] += len(selected_by_source['hub'])
        self.timing_stats['augmentation_filler_nodes'] += len(filler_nodes)

        final_nodes = torch.tensor(sorted(selected_set), dtype=torch.long)
        expected_size = int(core_partition.numel()) + target_extra
        if int(final_nodes.numel()) != expected_size:
            raise RuntimeError(
                f"Window augmentation size mismatch: strategy={self.window_aug_strategy}, "
                f"partition={partition_id}, expected={expected_size}, actual={final_nodes.numel()}"
            )
        return final_nodes, torch.empty((2, 0), dtype=torch.long)



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
        n_parts=n_parts,
        edge_index=edge_index,
        edge_csr_data=None,
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
