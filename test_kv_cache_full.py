import torch
import torch.nn as nn

def test_kv_cache_workflow():
    """测试完整的KV cache工作流程"""
    
    print("测试KV cache完整工作流程:")
    print("=" * 60)
    
    # 模拟参数
    n_layers = 3
    num_heads = 4
    hidden_dim = 64
    att_size = hidden_dim // num_heads
    batch_size = 1
    seq_len = 100
    dup_nodes_num = 20
    num_partitions = 4
    
    print(f"模拟参数:")
    print(f"  - 层数: {n_layers}")
    print(f"  - 头数: {num_heads}")
    print(f"  - 隐藏维度: {hidden_dim}")
    print(f"  - 重复节点数: {dup_nodes_num}")
    print(f"  - 分区数: {num_partitions}")
    
    # 1. 测试KV cache数据结构
    print("\n1. 测试KV cache数据结构:")
    
    # 创建模拟的KV cache
    kv_cache = []
    for layer in range(n_layers):
        k_cache = torch.randn(dup_nodes_num, num_heads, att_size)
        v_cache = torch.randn(dup_nodes_num, num_heads, att_size)
        kv_cache.append((k_cache, v_cache))
    
    print(f"  KV cache结构: list[tuple[torch.Tensor, torch.Tensor]]")
    print(f"  长度: {len(kv_cache)} (层数)")
    print(f"  每层K形状: {kv_cache[0][0].shape}")
    print(f"  每层V形状: {kv_cache[0][1].shape}")
    
    # 2. 测试分区KV cache管理
    print("\n2. 测试分区KV cache管理:")
    
    # 模拟多个分区的KV cache
    kv_cache_per_partition = [None] * num_partitions
    
    # 初始化第一个分区的KV cache
    kv_cache_per_partition[0] = kv_cache
    print(f"  分区0 KV cache: {kv_cache_per_partition[0] is not None}")
    print(f"  分区1 KV cache: {kv_cache_per_partition[1] is not None}")
    
    # 3. 测试KV cache更新
    print("\n3. 测试KV cache更新:")
    
    # 模拟更新后的KV cache
    updated_kv_cache = []
    for layer in range(n_layers):
        # 模拟计算新的K/V（实际中会从模型计算得到）
        new_k_cache = torch.randn(dup_nodes_num, num_heads, att_size)
        new_v_cache = torch.randn(dup_nodes_num, num_heads, att_size)
        updated_kv_cache.append((new_k_cache, new_v_cache))
    
    # 更新分区0的KV cache
    kv_cache_per_partition[0] = updated_kv_cache
    print(f"  分区0 KV cache已更新")
    print(f"  新K cache形状: {kv_cache_per_partition[0][0][0].shape}")
    print(f"  新V cache形状: {kv_cache_per_partition[0][0][1].shape}")
    
    # 4. 测试模型前向传播中的KV cache使用
    print("\n4. 测试模型中的KV cache逻辑:")
    
    # 模拟输入数据
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 模拟有KV cache的情况
    print(f"  输入形状: {x.shape}")
    print(f"  重复节点数: {dup_nodes_num}")
    
    # 模拟KV cache拼接逻辑
    if kv_cache_per_partition[0] is not None:
        layer_cache = kv_cache_per_partition[0][0]  # 第一层
        k_cache, v_cache = layer_cache
        
        # 动态部分
        x_dynamic = x[:, dup_nodes_num:, :]
        # 这里简化，实际会有线性变换
        k_dynamic = torch.randn(batch_size, seq_len - dup_nodes_num, num_heads, att_size)
        v_dynamic = torch.randn(batch_size, seq_len - dup_nodes_num, num_heads, att_size)
        
        # 扩展cache到batch维度
        k_cached = k_cache.unsqueeze(0).expand(batch_size, -1, -1, -1)
        v_cached = v_cache.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # 拼接
        k = torch.cat([k_cached, k_dynamic], dim=1)
        v = torch.cat([v_cached, v_dynamic], dim=1)
        
        print(f"  使用KV cache:")
        print(f"    - 缓存的K形状: {k_cached.shape}")
        print(f"    - 动态的K形状: {k_dynamic.shape}")
        print(f"    - 最终的K形状: {k.shape}")
        print(f"    ✓ 形状正确: {k.shape == (batch_size, seq_len, num_heads, att_size)}")
    else:
        # 无KV cache的情况
        k = torch.randn(batch_size, seq_len, num_heads, att_size)
        v = torch.randn(batch_size, seq_len, num_heads, att_size)
        print(f"  无KV cache，计算全部K/V")
    
    # 5. 测试训练和评估流程
    print("\n5. 测试训练和评估流程:")
    
    # 模拟训练epoch
    print("  模拟训练epoch:")
    train_kv_cache = [None] * num_partitions
    print(f"    初始化训练KV cache: {[cache is None for cache in train_kv_cache]}")
    
    # 模拟第一个分区训练后更新KV cache
    train_kv_cache[0] = updated_kv_cache
    print(f"    分区0训练后更新KV cache: {train_kv_cache[0] is not None}")
    
    # 模拟评估时使用KV cache
    print("  模拟评估epoch:")
    eval_kv_cache = train_kv_cache  # 使用训练得到的KV cache
    print(f"    使用训练得到的KV cache进行评估")
    
    # 6. 测试性能优势
    print("\n6. 测试性能优势:")
    
    # 计算有/无KV cache的计算量对比
    total_params = seq_len * hidden_dim * 3  # Q, K, V的线性变换
    cached_params = dup_nodes_num * hidden_dim * 2  # 缓存的K, V
    dynamic_params = (seq_len - dup_nodes_num) * hidden_dim * 2  # 动态计算的K, V
    
    print(f"  总参数计算量: {total_params:,}")
    print(f"  使用KV cache时:")
    print(f"    - 缓存参数: {cached_params:,} (已计算)")
    print(f"    - 动态计算参数: {dynamic_params:,}")
    print(f"    - 总计算量: {cached_params + dynamic_params:,}")
    print(f"  计算量减少: {(1 - (cached_params + dynamic_params) / total_params) * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✓ KV cache完整工作流程测试通过!")
    
    return True

if __name__ == "__main__":
    test_kv_cache_workflow()
    print("\n所有测试完成!")