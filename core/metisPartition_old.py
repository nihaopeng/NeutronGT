import torch
import pymetis

from utils.logger import log

class PartitionNode:
    def __init__(self, node_ids, pid, parent=None):
        self.pid = pid
        self.node_ids = node_ids          # 当前分区包含的节点
        self.children = []                # 子分区
        self.parent = parent              # 父分区引用

class PartitionTree:
    def __init__(self, feature, edge_index, train_idx, max_depth=2, n_parts=2):
        self.root = None
        self.tree = {}
        self.partition_counter = 0
        self.max_depth = max_depth
        self.n_parts = n_parts
        self.feature = feature
        self.edge_index = edge_index
        self.train_idx = train_idx

    def _new_partition(self, node_ids, parent=None):
        pid = self.partition_counter
        self.partition_counter += 1
        node = PartitionNode(node_ids, pid, parent)
        self.tree[pid] = node
        if parent is not None:
            parent.children.append(node)
        else:
            self.root = node
        return node

    def _build_adjacency(self, edge_index, node_ids):
        id_map = {nid: i for i, nid in enumerate(node_ids)}
        adjacency = [[] for _ in node_ids]
        for u, v in edge_index.t().tolist():
            if u in id_map and v in id_map:
                adjacency[id_map[u]].append(id_map[v])
                adjacency[id_map[v]].append(id_map[u])
        return adjacency

    def _recursive_partition(self, node, edge_index, depth):
        if depth >= self.max_depth or len(node.node_ids) <= self.n_parts:
            return
        adjacency = self._build_adjacency(edge_index, node.node_ids)
        if not any(adjacency):
            return
        _, parts = pymetis.part_graph(self.n_parts, adjacency=adjacency)
        partitions = [[] for _ in range(self.n_parts)]
        for i, pid in enumerate(parts):
            partitions[pid].append(node.node_ids[i])
        for part_nodes in partitions:
            if part_nodes:
                child = self._new_partition(part_nodes, parent=node)
                self._recursive_partition(child, edge_index, depth+1)

    def partition_nodes_metis(self):
        num_nodes = self.feature.size(0)
        root_node = self._new_partition(list(range(num_nodes)))
        self._recursive_partition(root_node, self.edge_index, depth=0)
        return self.root

    def get_final_partitions(self):
        """
        获取最终分区（树的叶子节点）及其对应的特征
        
        参数:
            train_idx: torch.Tensor, 训练节点的索引。如果提供，则只返回包含这些索引的分区
        
        返回:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[PartitionNode]] - 
                过滤后的分区节点索引列表、对应的特征张量列表和分区节点对象列表
        """
        final_parts = []
        final_features = []
        final_nodes = []
        
        def dfs(node):
            if not node.children:  # 叶子节点
                node_indices = torch.tensor(node.node_ids, dtype=torch.long)
                
                # 如果提供了train_idx，则过滤出在train_idx中的节点
                if self.train_idx is not None:
                    # 创建掩码，标记哪些节点在train_idx中
                    mask = torch.isin(node_indices, self.train_idx)
                    filtered_indices = node_indices[mask]
                    
                    # 只有当分区中至少有一个训练节点时才保留该分区
                    if len(filtered_indices) > 0:
                        # 获取对应的特征
                        part_features = self.feature[filtered_indices]
                        final_parts.append(filtered_indices)
                        final_features.append(part_features)
                        final_nodes.append(node)
                else:
                    # 如果没有提供train_idx，返回所有分区
                    part_features = self.feature[node_indices]
                    final_parts.append(node_indices)
                    final_features.append(part_features)
                    final_nodes.append(node)
            else:
                for child in node.children:
                    dfs(child)

        dfs(self.root)
        return final_parts, final_features, final_nodes

    def dynamic_windows_globalReassign(self, scores_partitions, nodes,*args,**kwargs):
        """
        全局淘汰池：收集所有分区的淘汰节点，再重新分配到各个窗口
        不返回任何值，直接更新树结构
        """
        global_removed = []
        kept_dict = {}

        # 第一步：收集淘汰节点
        for node, scores in zip(nodes, scores_partitions):
            part_nodes = torch.tensor(node.node_ids)
            part_nodes = part_nodes[torch.isin(part_nodes, self.train_idx)]
            log(f"score shape:{scores.shape}")
            row_sums = scores.abs().sum(dim=1)
            k = int(kwargs["remove_ratio"] * len(row_sums))
            _, indices = torch.topk(row_sums, k, largest=False)
            keep_mask = torch.ones_like(row_sums, dtype=torch.bool).to(part_nodes.device)
            keep_mask[indices] = False
            # keep_mask = scores.abs().sum(dim=1) < 0.001  # TODO:阈值计算
            log(f"node:{part_nodes.shape}, scores:{scores.shape}, mask:{keep_mask.shape}")
            
            kept_nodes = part_nodes[keep_mask]
            removed_nodes = part_nodes[~keep_mask]

            kept_dict[node.pid] = kept_nodes
            global_removed.extend(removed_nodes.tolist())

        global_removed = torch.tensor(global_removed, dtype=torch.long)
        log(f"global_removed:\n{global_removed}")
        
        # 第二步：重新分配淘汰节点
        idx = 0
        alloc_num = len(global_removed) // len(nodes)
        for node in nodes:
            kept_nodes = kept_dict[node.pid]
            num_needed = len(node.node_ids) - len(kept_nodes)
            
            # 简单策略：顺序分配
            supplement = global_removed[idx: idx + alloc_num]
            idx += alloc_num

            new_nodes = torch.cat([kept_nodes, supplement])
            node.node_ids = new_nodes.tolist()
            
            print(f"分区 {node.pid}: 保留 {len(kept_nodes)} 个节点，补充 {len(supplement)} 个节点")

        # 第三步：更新父节点
        for node in nodes:
            self._update_parent(node.parent)
        
        # 不返回任何值，树结构已更新

    def _update_parent(self, parent_node):
        """
        根据子分区更新父分区的节点集合
        """
        if parent_node is None:
            return
        
        # 收集所有子节点的节点ID
        merged = []
        for child in parent_node.children:
            merged.extend(child.node_ids)
        
        # 去重并排序
        parent_node.node_ids = list(sorted(set(merged)))
        
        # 递归更新上级父节点
        self._update_parent(parent_node.parent)

    def dynamic_windows_metisParent(self,scores_partitions, nodes,*args,**kwargs):
        """
        使用metis分区，查询父分区进行重构
        """
        new_partitions = []
        for node, scores in zip(nodes, scores_partitions):
            part_nodes = torch.tensor(node.node_ids)
            keep_mask = scores.sum(dim=1) > 0
            kept_nodes = part_nodes[keep_mask]
            removed_nodes = part_nodes[~keep_mask]

            # 从父分区中寻找替补
            parent_nodes = torch.tensor(node.parent.node_ids)
            candidates = parent_nodes[~torch.isin(parent_nodes, part_nodes)]
            num_needed = len(removed_nodes)
            supplement = candidates[:num_needed]

            new_nodes = torch.cat([kept_nodes, supplement])
            node.node_ids = new_nodes.tolist()
            new_partitions.append(new_nodes)

        return new_partitions

    def dynamic_window_build(self, scores_partitions, nodes, mode="globalReassign",*args,**kwargs):
        """
        scores_partitions:
        nodes:
        动态窗口更新: 从父分区中寻找被踢出的节点，重新分配到合适窗口
        """
        if mode=="metisParents":
            return self.dynamic_windows_metisParent(scores_partitions,nodes,*args,**kwargs)
        elif mode=="globalReassign":
            return self.dynamic_windows_globalReassign(scores_partitions,nodes,*args,**kwargs)

if __name__=="__main__":
    import os
    # dataset_dir = "/home/ma-user/work/projects/torchgt_npu/dataset/ogbn-arxiv"
    # feature = torch.load(os.path.join(dataset_dir, 'x.pt')) # [N, x_dim]
    # y = torch.load(os.path.join(dataset_dir, 'y.pt')) # [N]
    # edge_index = torch.load(os.path.join(dataset_dir, 'edge_index.pt')) # [2, num_edges]
    # partitionTree = PartitionTree(feature,edge_index,3)
    # partitionTree.partition_nodes_metis()
    # parts = partitionTree.get_final_partitions()
    # for part in parts:
    #     print(f"parts:{part.shape}")
    # ====== 构造一个简单的图 ======
    # 节点特征 (10个节点，每个节点2维特征)
    feature = torch.randn(10, 2)
    log(f"feature:\n{feature}")

    # 边索引 (无向图)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 2, 4, 6, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
        2, 4, 6, 8, 0]
    ], dtype=torch.long)

    train_idx = torch.tensor([2,3,4,5,6])
    
    # ====== 构建分区树 ======
    tree = PartitionTree(feature, edge_index, train_idx, max_depth=2, n_parts=2)
    root = tree.partition_nodes_metis()

    # ====== 获取最终分区（叶子节点） ======
    final_parts,final_feature,final_nodes = tree.get_final_partitions()
    log("初始分区结果：")
    for i, part in enumerate(final_parts):
        log(f"Partition {i}: {part.tolist()}")
        log(f"feature {i}: {final_feature[i]}")

    # ====== 构造模拟的分数矩阵 ======
    # 每个分区一个分数矩阵，大小 = 分区节点数 × 分区节点数
    scores_partitions = []
    for part in final_parts:
        n = len(part)
        # 随机分数，部分节点可能被淘汰（行和为负）
        scores = torch.randn(n, n)
        scores_partitions.append(scores)
    log("scores:")
    for scores in scores_partitions:
        log(scores)
    # ====== 调用动态窗口更新 ======
    tree.dynamic_window_build(scores_partitions, nodes=final_nodes, mode="globalReassign")
    # print(new_partitions)
    new_parts,new_feature,new_nodes = tree.get_final_partitions()
    log("\n动态更新后的分区结果：")
    for i, part in enumerate(new_parts):
        log(f"Partition {i}: {part.tolist()}")