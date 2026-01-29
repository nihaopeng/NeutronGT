import torch
import numpy as np

def simulate_partition_stats():
    """模拟分区数据并计算重复节点统计"""
    
    print("模拟分区重复节点统计:")
    print("="*80)
    
    # 模拟参数
    total_nodes = 1000
    num_partitions = 10
    partition_size = 200  # 每个分区大小
    overlap_prob = 0.3   # 节点出现在多个分区的概率
    
    print(f"模拟参数:")
    print(f"  - 总节点数: {total_nodes}")
    print(f"  - 分区数: {num_partitions}")
    print(f"  - 每个分区大小: {partition_size}")
    print(f"  - 节点重叠概率: {overlap_prob*100}%")
    
    # 生成模拟数据
    np.random.seed(42)
    
    # 生成分区
    partitions = []
    for i in range(num_partitions):
        # 随机选择节点，允许重复
        nodes = np.random.choice(total_nodes, size=partition_size, replace=False)
        partitions.append(torch.tensor(nodes, dtype=torch.long))
    
    # 计算重复节点
    all_nodes = torch.cat(partitions)
    unique_nodes, counts = torch.unique(all_nodes, return_counts=True)
    duplicated_nodes = unique_nodes[counts >= 2]
    
    # 统计每个分区的重复节点
    dup_nodes_per_partition = []
    for partition in partitions:
        dup_mask = torch.isin(partition, duplicated_nodes)
        dup_nodes = partition[dup_mask]
        dup_nodes_per_partition.append(dup_nodes)
    
    # 1. 全局重复节点统计
    total_dup_nodes = len(duplicated_nodes)
    dup_ratio_global = total_dup_nodes / total_nodes * 100
    
    print(f"\n全局统计:")
    print(f"  - 总节点数: {total_nodes}")
    print(f"  - 重复节点数: {total_dup_nodes}")
    print(f"  - 重复节点占比: {dup_ratio_global:.2f}%")
    
    # 2. 各分区重复节点统计
    print(f"\n各分区统计:")
    total_partition_nodes = 0
    total_dup_in_partitions = 0
    
    for i, (partition, dup_nodes) in enumerate(zip(partitions, dup_nodes_per_partition)):
        partition_size = len(partition)
        dup_size = len(dup_nodes)
        dup_ratio = dup_size / partition_size * 100 if partition_size > 0 else 0
        
        total_partition_nodes += partition_size
        total_dup_in_partitions += dup_size
        
        print(f"  分区 {i:2d}: 总节点={partition_size:4d}, 重复节点={dup_size:4d}, 占比={dup_ratio:6.2f}%")
    
    # 3. 分区重叠度统计
    print(f"\n分区重叠度统计:")
    node_partition_count = {}
    for i, partition in enumerate(partitions):
        for node in partition.tolist():
            node_partition_count[node] = node_partition_count.get(node, 0) + 1
    
    partition_counts = list(node_partition_count.values())
    if partition_counts:
        avg_overlap = sum(partition_counts) / len(partition_counts)
        max_overlap = max(partition_counts)
        min_overlap = min(partition_counts)
        
        print(f"  - 平均每个节点出现在 {avg_overlap:.2f} 个分区中")
        print(f"  - 最大重叠度: {max_overlap} 个分区")
        print(f"  - 最小重叠度: {min_overlap} 个分区")
        
        # 重叠度分布
        overlap_dist = {}
        for count in partition_counts:
            overlap_dist[count] = overlap_dist.get(count, 0) + 1
        
        print(f"  - 重叠度分布:")
        for count in sorted(overlap_dist.keys()):
            ratio = overlap_dist[count] / len(partition_counts) * 100
            print(f"     出现在 {count} 个分区: {overlap_dist[count]:4d} 个节点 ({ratio:5.1f}%)")
    
    # 4. 计算效率提升预估
    print(f"\n效率提升预估:")
    compute_reduction = dup_ratio_global / 100 * 100
    print(f"  - 理论计算量减少: {compute_reduction:.1f}%")
    print(f"  - 理论加速比: {1/(1-compute_reduction/100):.2f}x")
    
    # 5. 可视化数据
    print(f"\n可视化分析:")
    
    # 计算分区间的相似度（Jaccard相似度）
    print(f"  - 分区间相似度矩阵 (前5个分区):")
    for i in range(min(5, num_partitions)):
        row = []
        for j in range(min(5, num_partitions)):
            set_i = set(partitions[i].tolist())
            set_j = set(partitions[j].tolist())
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            similarity = intersection / union if union > 0 else 0
            row.append(f"{similarity:.3f}")
        print(f"    分区{i}: [{', '.join(row)}]")
    
    # 6. 重复节点特征分析
    print(f"\n重复节点特征分析:")
    if len(duplicated_nodes) > 0:
        # 计算重复节点的平均出现次数
        avg_dup_count = counts[counts >= 2].float().mean().item()
        max_dup_count = counts.max().item()
        
        print(f"  - 重复节点平均出现在 {avg_dup_count:.2f} 个分区中")
        print(f"  - 最多出现在 {max_dup_count} 个分区中")
        
        # 重复节点出现次数分布
        dup_count_dist = {}
        for count in counts[counts >= 2]:
            dup_count_dist[int(count)] = dup_count_dist.get(int(count), 0) + 1
        
        print(f"  - 重复节点出现次数分布:")
        for count in sorted(dup_count_dist.keys()):
            print(f"     出现{count}次: {dup_count_dist[count]:4d} 个节点")
    
    print("="*80)
    print("✓ 重复节点统计模拟完成!")
    
    return partitions, dup_nodes_per_partition

if __name__ == "__main__":
    simulate_partition_stats()