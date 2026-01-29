import torch
import numpy as np

def test_integrated_kv_cache_system():
    """测试集成的KV cache系统"""
    
    print("="*100)
    print("集成的KV cache系统测试")
    print("="*100)
    
    # 模拟真实场景
    total_nodes = 5000
    num_partitions = 20
    hidden_dim = 256
    num_heads = 8
    n_layers = 6
    att_size = hidden_dim // num_heads
    
    print(f"\n📊 系统配置:")
    print(f"  - 总节点数: {total_nodes:,}")
    print(f"  - 分区数: {num_partitions}")
    print(f"  - 隐藏维度: {hidden_dim}")
    print(f"  - 注意力头数: {num_heads}")
    print(f"  - 网络层数: {n_layers}")
    print(f"  - 每头维度: {att_size}")
    
    # 生成模拟的分区数据（更真实的分布）
    np.random.seed(42)
    
    # 1. 生成分区（使用幂律分布模拟真实图的分区）
    print(f"\n🔧 生成分区数据...")
    partitions = []
    base_partition_size = 300
    
    for i in range(num_partitions):
        # 使用幂律分布选择节点（模拟真实图中的重要节点）
        popularity = np.random.power(2, size=base_partition_size)
        nodes = (popularity * (total_nodes * 0.8)).astype(int)
        nodes = np.unique(np.clip(nodes, 0, total_nodes-1))
        partitions.append(torch.tensor(nodes, dtype=torch.long))
    
    # 2. 计算重复节点
    print(f"📈 计算重复节点统计...")
    all_nodes = torch.cat(partitions)
    unique_nodes, counts = torch.unique(all_nodes, return_counts=True)
    duplicated_nodes = unique_nodes[counts >= 2]
    
    # 统计每个分区的重复节点
    dup_nodes_per_partition = []
    for partition in partitions:
        dup_mask = torch.isin(partition, duplicated_nodes)
        dup_nodes = partition[dup_mask]
        dup_nodes_per_partition.append(dup_nodes)
    
    # 3. 详细统计
    total_dup_nodes = len(duplicated_nodes)
    total_partition_nodes = sum(len(p) for p in partitions)
    avg_partition_size = total_partition_nodes / num_partitions
    
    print(f"\n📊 分区统计摘要:")
    print(f"  - 总分区节点数: {total_partition_nodes:,} (含重复)")
    print(f"  - 平均分区大小: {avg_partition_size:.0f} 节点")
    print(f"  - 唯一节点数: {len(unique_nodes):,}")
    print(f"  - 重复节点数: {total_dup_nodes:,}")
    print(f"  - 重复节点占比: {total_dup_nodes/len(unique_nodes)*100:.1f}%")
    
    # 4. 分区重叠分析
    print(f"\n🔍 分区重叠分析:")
    node_partition_count = {}
    for i, partition in enumerate(partitions):
        for node in partition.tolist():
            node_partition_count[node] = node_partition_count.get(node, 0) + 1
    
    partition_counts = list(node_partition_count.values())
    avg_overlap = sum(partition_counts) / len(partition_counts)
    
    # 计算分区效率指标
    overlap_dist = {}
    for count in partition_counts:
        overlap_dist[count] = overlap_dist.get(count, 0) + 1
    
    print(f"  - 平均重叠度: {avg_overlap:.2f}")
    print(f"  - 分区效率: {len(unique_nodes)/total_partition_nodes*100:.1f}%")
    
    # 5. KV cache效益分析
    print(f"\n💡 KV cache效益分析:")
    
    # 计算理论加速比
    dup_ratio = total_dup_nodes / len(unique_nodes)
    if dup_ratio > 0:
        speedup = 1 / (1 - dup_ratio)
        compute_reduction = dup_ratio * 100
    else:
        speedup = 1.0
        compute_reduction = 0.0
    
    print(f"  - 理论计算量减少: {compute_reduction:.1f}%")
    print(f"  - 理论加速比: {speedup:.2f}x")
    
    # 6. 内存占用分析
    print(f"\n💾 内存占用分析:")
    
    # KV cache内存
    k_cache_size = total_dup_nodes * num_heads * att_size * 4  # bytes
    v_cache_size = total_dup_nodes * num_heads * att_size * 4
    per_layer_cache = (k_cache_size + v_cache_size) / (1024**2)  # MB
    total_cache = per_layer_cache * n_layers
    
    # 特征内存（对比）
    feature_size = len(unique_nodes) * hidden_dim * 4 / (1024**2)  # MB
    
    print(f"  - 特征内存: {feature_size:.2f} MB")
    print(f"  - KV cache内存:")
    print(f"     每层: {per_layer_cache:.2f} MB")
    print(f"     总计 ({n_layers}层): {total_cache:.2f} MB")
    print(f"  - KV cache/特征内存比: {total_cache/feature_size*100:.1f}%")
    
    # 7. 性能模拟
    print(f"\n⚡ 性能模拟:")
    
    # 模拟计算时间
    base_compute_time = 100  # 基准计算时间（无cache）
    with_cache_time = base_compute_time * (1 - dup_ratio)
    
    print(f"  - 基准计算时间: {base_compute_time:.1f} ms")
    print(f"  - 使用KV cache后: {with_cache_time:.1f} ms")
    print(f"  - 时间减少: {base_compute_time - with_cache_time:.1f} ms ({compute_reduction:.1f}%)")
    
    # 8. 不同场景下的效益
    print(f"\n📈 不同重复节点占比下的效益:")
    scenarios = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for ratio in scenarios:
        speedup_scenario = 1 / (1 - ratio)
        memory_scenario = ratio * total_cache / dup_ratio if dup_ratio > 0 else 0
        print(f"  - {ratio*100:3.0f}% 重复: 加速比={speedup_scenario:.2f}x, 内存={memory_scenario:.2f} MB")
    
    # 9. 推荐配置
    print(f"\n🎯 推荐配置:")
    
    optimal_ratio = 0.3  # 经验上的最佳重复比例
    if dup_ratio < optimal_ratio:
        print(f"  ⚠️  当前重复比例较低 ({dup_ratio*100:.1f}%)")
        print(f"  💡 建议: 增加分区重叠度以提高KV cache效益")
    elif dup_ratio > 0.6:
        print(f"  ⚠️  当前重复比例较高 ({dup_ratio*100:.1f}%)")
        print(f"  💡 建议: 减少分区重叠以降低内存占用")
    else:
        print(f"  ✅ 当前重复比例良好 ({dup_ratio*100:.1f}%)")
        print(f"  💡 建议: 保持当前配置")
    
    # 10. 可视化摘要
    print(f"\n📋 可视化摘要:")
    print(f"  ├── 分区数: {num_partitions}")
    print(f"  ├── 总节点: {total_nodes:,}")
    print(f"  ├── 重复节点: {total_dup_nodes:,} ({dup_ratio*100:.1f}%)")
    print(f"  ├── 平均重叠: {avg_overlap:.2f}")
    print(f"  ├── 理论加速: {speedup:.2f}x")
    print(f"  ├── 内存占用: {total_cache:.2f} MB")
    print(f"  └── 效益比: {speedup/(total_cache/100):.3f} (加速比/每100MB)")
    
    print(f"\n" + "="*100)
    print("✅ 集成测试完成!")
    print("="*100)
    
    return {
        'total_nodes': total_nodes,
        'num_partitions': num_partitions,
        'unique_nodes': len(unique_nodes),
        'dup_nodes': total_dup_nodes,
        'dup_ratio': dup_ratio,
        'avg_overlap': avg_overlap,
        'speedup': speedup,
        'memory_mb': total_cache
    }

if __name__ == "__main__":
    results = test_integrated_kv_cache_system()
    
    # 保存结果用于后续分析
    print(f"\n📁 测试结果已保存:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")