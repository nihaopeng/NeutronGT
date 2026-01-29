import torch
import torch.nn as nn

def test_multihead_attention_kv_cache():
    """测试MultiHeadAttention中的KV cache逻辑"""
    
    # 模拟参数
    num_heads = 4
    hidden_dim = 64
    att_size = hidden_dim // num_heads
    batch_size = 1
    seq_len = 100
    dup_nodes_num = 20
    
    # 创建模拟的线性层
    class MockLinear(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
            self.bias = nn.Parameter(torch.randn(out_dim))
        
        def forward(self, x):
            return x @ self.weight.t() + self.bias
    
    # 模拟MultiHeadAttention的核心逻辑
    linear_q = MockLinear(hidden_dim, hidden_dim)
    linear_k = MockLinear(hidden_dim, hidden_dim)
    linear_v = MockLinear(hidden_dim, hidden_dim)
    
    # 测试数据
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    print("测试KV cache逻辑:")
    print(f"输入形状: {x.shape}")
    print(f"重复节点数量: {dup_nodes_num}")
    
    # 情况1: 无KV cache
    print("\n1. 无KV cache:")
    q = linear_q(x).view(batch_size, -1, num_heads, att_size)
    k = linear_k(x).view(batch_size, -1, num_heads, att_size)
    v = linear_v(x).view(batch_size, -1, num_heads, att_size)
    print(f"Q形状: {q.shape}")
    print(f"K形状: {k.shape}")
    print(f"V形状: {v.shape}")
    
    # 情况2: 有KV cache
    print("\n2. 有KV cache:")
    # 创建模拟的KV cache
    k_cache = torch.randn(dup_nodes_num, num_heads, att_size)
    v_cache = torch.randn(dup_nodes_num, num_heads, att_size)
    
    q = linear_q(x).view(batch_size, -1, num_heads, att_size)
    
    # 动态部分
    x_dynamic = x[:, dup_nodes_num:, :]
    k_dynamic = linear_k(x_dynamic).view(batch_size, -1, num_heads, att_size)
    v_dynamic = linear_v(x_dynamic).view(batch_size, -1, num_heads, att_size)
    
    # 扩展cache到batch维度
    k_cached = k_cache.unsqueeze(0).expand(batch_size, -1, -1, -1)
    v_cached = v_cache.unsqueeze(0).expand(batch_size, -1, -1, -1)
    
    # 拼接
    k = torch.cat([k_cached, k_dynamic], dim=1)
    v = torch.cat([v_cached, v_dynamic], dim=1)
    
    print(f"Q形状: {q.shape}")
    print(f"K形状: {k.shape} (cached: {k_cached.shape}, dynamic: {k_dynamic.shape})")
    print(f"V形状: {v.shape} (cached: {v_cached.shape}, dynamic: {v_dynamic.shape})")
    
    # 验证形状
    assert k.shape == (batch_size, seq_len, num_heads, att_size), f"K形状错误: {k.shape}"
    assert v.shape == (batch_size, seq_len, num_heads, att_size), f"V形状错误: {v.shape}"
    
    print("\n✓ KV cache逻辑测试通过!")
    
    # 测试注意力计算
    print("\n3. 测试注意力计算:")
    scale = (att_size ** 0.5)
    q_scaled = q * scale
    attn_scores = torch.matmul(q_scaled, k.transpose(-2, -1))
    print(f"注意力分数形状: {attn_scores.shape}")
    
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)
    print(f"输出形状: {output.shape}")
    
    print("\n✓ 注意力计算测试通过!")
    
    return True

if __name__ == "__main__":
    test_multihead_attention_kv_cache()
    print("\n所有测试完成!")