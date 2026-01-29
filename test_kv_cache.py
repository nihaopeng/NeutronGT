import torch
import torch.nn as nn
from models.gt_dist_node_level_single_window import GT_SW

def test_kv_cache():
    # 模拟参数
    n_layers = 3
    num_heads = 4
    hidden_dim = 64
    att_size = hidden_dim // num_heads
    batch_size = 1
    seq_len = 100
    dup_nodes_num = 20  # 重复节点数量
    
    # 创建模拟数据
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建模型
    class MockArgs:
        def __init__(self):
            self.n_layers = n_layers
            self.hidden_dim = hidden_dim
            self.struct_enc = "False"
            self.use_cache = 1
    
    args = MockArgs()
    
    model = GT_SW(
        n_layers=n_layers,
        num_heads=num_heads,
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        output_dim=10,
        attn_bias_dim=0,
        dropout_rate=0.1,
        input_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        ffn_dim=hidden_dim * 2,
        num_global_node=0,
        args=args,
        num_in_degree=0,
        num_out_degree=0,
        num_spatial=10,  # 需要大于0
        num_edges=10,
        max_dist=5,
        edge_dim=8
    )
    
    # 测试1: 无KV cache的情况
    print("测试1: 无KV cache")
    output1, _, _, kv_cache1 = model(
        x.squeeze(0),  # 去掉batch维度
        attn_bias=None,
        edge_index=None,
        in_degree=None,
        out_degree=None,
        spatial_pos=None,
        edge_input=None,
        attn_type="full",
        dup_nodes_kv_cache=None,
        part_id=0
    )
    print(f"输出形状: {output1.shape}")
    print(f"KV cache: {kv_cache1}")
    
    # 测试2: 有KV cache的情况
    print("\n测试2: 有KV cache")
    # 创建模拟的KV cache
    mock_kv_cache = []
    for _ in range(n_layers):
        k_cache = torch.randn(dup_nodes_num, num_heads, att_size)
        v_cache = torch.randn(dup_nodes_num, num_heads, att_size)
        mock_kv_cache.append((k_cache, v_cache))
    
    output2, _, _, kv_cache2 = model(
        x.squeeze(0),
        attn_bias=None,
        edge_index=None,
        in_degree=None,
        out_degree=None,
        spatial_pos=None,
        edge_input=None,
        attn_type="full",
        dup_nodes_kv_cache=mock_kv_cache,
        part_id=0
    )
    print(f"输出形状: {output2.shape}")
    print(f"KV cache类型: {type(kv_cache2)}")
    if kv_cache2 is not None:
        print(f"KV cache长度: {len(kv_cache2)}")
        if len(kv_cache2) > 0 and kv_cache2[0] is not None:
            print(f"第一层K cache形状: {kv_cache2[0][0].shape if kv_cache2[0][0] is not None else 'None'}")
            print(f"第一层V cache形状: {kv_cache2[0][1].shape if kv_cache2[0][1] is not None else 'None'}")
    
    # 测试3: 检查输出是否一致
    print("\n测试3: 检查输出一致性")
    if output1.shape == output2.shape:
        diff = torch.abs(output1 - output2).max().item()
        print(f"最大差异: {diff}")
        if diff < 1e-5:
            print("✓ 输出基本一致")
        else:
            print("⚠ 输出有差异，可能是KV cache的影响")
    else:
        print("✗ 输出形状不一致")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_kv_cache()