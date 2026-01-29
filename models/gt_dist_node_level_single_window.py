import time
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from gt_sp.gt_layer import DistributedAttentionNodeLevel, _SeqGather
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_global_token_indices,
)
from torch_scatter import scatter

from utils.logger import log

# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        
class CoreAttention(nn.Module):
    """
    Core attn 
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(CoreAttention, self).__init__()

        # SP group: Per attention head and per partition values.
        seq_parallel_world_size = 1
        if sequence_parallel_is_initialized():
            seq_parallel_world_size = get_sequence_parallel_world_size()
        world_size = seq_parallel_world_size 

        self.hidden_size_per_partition = hidden_size // world_size
        self.hidden_size_per_attention_head =  hidden_size // num_heads
        self.num_attention_heads_per_partition = num_heads // world_size

        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.num_heads = num_heads
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.attention_dropout_rate = attention_dropout_rate

    # def full_flash_attention(self, q, k, v, attn_bias=None, mask=None):
    #     return flash_attn_func(q, k, v, self.attention_dropout_rate)
    

    def full_attention(self, k, q, v, attn_bias, mask=None,pruning_mask=None):
        # ===================================
        # qkv.                  [b, seq_len, num_head, atten_size]
        # atten_size denote dim per head 
        # Raw Attn Score        [b, num_head, seq_len, seq_len]
        # attn_bias.            [b, num_head, seq_len, seq_len]
        # ===================================
        log(f"q:{q.shape},k:{k.shape},v:{v.shape}")
        q = q.transpose(1, 2)                    # [b, num_head, seq_len, atten_size]
        v = v.transpose(1, 2)                    # [b, num_head, seq_len, atten_size]
        k = k.transpose(1, 2).transpose(2, 3)    # [b, num_head, atten_size, seq_len]
        log(f"transpose->q:{q.shape},k:{k.shape},v:{v.shape}")
        
        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        # 
        # Q * K^T:    [b, num_head, seq_len, atten_size] * [b, num_head, atten_size, seq_len] = [b, num_head, seq_len, seq_len]
        # score * V:  [b, num_head, seq_len, seq_len] * [b, num_head, seq_len, atten_size] = [b, num_head, seq_len, atten_size]
        
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, num_head, seq_len, seq_len]
        log(f"x shape:{x.shape}")
        score = x
        if attn_bias is not None:
            # attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1)
            x = x + attn_bias
            
            
        # =============采样掩码===========    
        if pruning_mask is not None:
            x = x + pruning_mask
        # ===============================    
            
        if mask is not None:
            # print(f"mask shape:{mask.shape}")
            mask = mask.to(x.device)
            mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(0)
            mask = mask.repeat(1, x.shape[1], 1, 1)
            # print(f"mask shape:{mask.shape}")
            # print(f"mask sum:{torch.sum(mask)}")
            # print(f"x shape:{x.shape}")
            # input()
            x = x.masked_fill(mask, -1e9) # 核心修改：0 → -1e9,softmax后才会真正变为零
        # score = x

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        log(f"x shape:{x.shape}")
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        log(f"x shape:{x.shape}")
        return x,torch.abs(score)# return absoluted value # 对分数的绝对值求和，考虑正负数都起作用
    

    def sparse_attention_bias(self, q, k, v, edge_index, attn_bias):
        # q, k, v: [b, s, np, hn]  e: [total_edges, n, hn], edge_index: [2, total_edges], attn_bias: [b, n, s+1, s+1]
        batch_size, node_num = k.size(0), k.size(1)
        if self.training:
            num_heads = self.num_attention_heads_per_partition
        else:
            num_heads = self.num_heads
        
        # Reshaping into [total_s, np, hn] to
        # get projections for multi-head attention
        # kqv: [total_s, np, hn],  e: [total_edges, np, hn]
        # 这里的b恒为1，bs=s，所以索引不会出问题。
        q = q.view(-1, num_heads, self.hidden_size_per_attention_head)
        k = k.view(-1, num_heads, self.hidden_size_per_attention_head)
        v = v.view(-1, num_heads, self.hidden_size_per_attention_head)

        # -> [total_edges, np, hn]
        src = k[edge_index[0].to(torch.long)] 
        dest = q[edge_index[1].to(torch.long)]
        score = torch.mul(src, dest)  # element-wise multiplication
        
        # Scale scores by sqrt(d)
        score = score / self.scale

        # Use available edge features to modify the scores for edges
        # -> [total_edges, np, 1] 
        score = score.sum(-1, keepdim=True).clamp(-5, 5)

        # [b, np, s+1, s+1] -> [b, s+1, s+1, np] -> [b, s+1, b, s+1, np]
        if attn_bias is not None:
            # print(f"attnbias:{attn_bias.shape}")
            attn_bias = attn_bias.permute(0, 2, 3, 1).contiguous().unsqueeze(2).repeat(1, 1, batch_size, 1, 1) 
            # print(f"attnbias:{attn_bias.shape}")
            attn_bias = attn_bias.view(batch_size*node_num, batch_size*node_num, num_heads)
            # print(f"attnbias:{attn_bias.shape}")
            # attn_bias = attn_bias.repeat(1, 1, 1, num_heads)
            # print(f"score:{score.shape},attn_bias:{attn_bias.shape},index:{edge_index[0].to(torch.long).shape, edge_index[1].to(torch.long).shape}")
            # print(f"score:{score.shape},attn_bias:{attn_bias.shape},index:{attn_bias[edge_index[0].to(torch.long), edge_index[1].to(torch.long), :].unsqueeze(2) }")
            score = score + \
                    attn_bias[edge_index[0].to(torch.long), edge_index[1].to(torch.long), :].unsqueeze(2) 

        # softmax -> [total_edges, np, 1]
        # print(score[80:150, :2, 0])
        score = torch.exp(score) 
        # print(score[80:150, :2, 0])

        # Apply attention score to each source node to create edge messages
        # -> [total_edges, np, hn]
        msg = v[edge_index[0].to(torch.long)] * score
        # print(msg[110:150, :2, 0])
        # exit(0)
        
        # Add-up real msgs in destination nodes as given by edge_index[1]
        # -> [total_s, np, hn]
        wV = torch.zeros_like(v)  
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        # Compute attention normalization coefficient
        # -> [total_s, np, 1]
        Z = score.new_zeros(v.size(0), num_heads, 1)    
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

        x = wV / (Z + 1e-6)
        
        return x,score

    def naive_attention(self, q, k, v, dropout_p=0.0):
        # q, k, v: [batch, n_heads, seq_len, head_dim]
        d_k = q.size(-1)

        # 1. 计算注意力分数 (scaled dot-product)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

        # 2. softmax 归一化
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 3. dropout
        attn_probs = F.dropout(attn_probs, p=dropout_p, training=True)

        # 4. 加权求和
        output = torch.matmul(attn_probs, v)
        return output,None

    def forward(self, q, k, v, attn_bias=None, edge_index=None, attn_type=None,mask= None,pruning_mask=None):
        # ===================================
        # Raw attention scores. [b, np, s+1, s+1]
        # ===================================
        # q, k, v: [b, s+p, np, hn], edge_index: [2, total_edges], attn_bias: [b, n, s+p, s+p]
        batch_size, s_len = q.size(0), q.size(1)
        score = None
        if attn_type == "full":
            x,score = self.full_attention(k, q, v, attn_bias,pruning_mask=pruning_mask,mask=mask)
        elif attn_type == "sparse":
            # 这个sparse的score还不清楚是否可以
            # x,score = self.sparse_attention_bias(q, k, v, edge_index, attn_bias)
            x,score = self.sparse_attention_bias(q, k, v, edge_index, attn_bias)
        elif attn_type == "flash":
            q = q.half()
            k = k.half()
            v = v.half()
            # x = flash_attn_func(q, k, v, self.attention_dropout_rate)
            x = self.naive_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), dropout_p=self.attention_dropout_rate)
            x = x.float()
        
        # [b, s+p, hp]
        log(f"x:{x.shape}")
        x = x.view(batch_size, s_len, -1)
        log(f"x:{x.shape}")
        return x,score


class MultiHeadAttention(nn.Module):
    """Distributed multi-headed attention.

    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads # hn
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)

        self.local_attn = CoreAttention(
            hidden_size, attention_dropout_rate, num_heads)
        # ======================================= 取消跨设备分head操作，每个设备独立计算自己的窗口。
        self.dist_attn = DistributedAttentionNodeLevel(self.local_attn, get_sequence_parallel_group())


    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None,mask = None,pruning_mask=None,dup_nodes_kv_cache=None,layer:int=0):
        # x: [b, seq_len, hidden_dim],    attn_bias: [b, num_head, seq_len, seq_len]

        orig_q_size = x.size()
        batch_size, seq_len, hidden_dim = x.shape
        # =====================
        # Query, Key, and Value
        # =====================

        # x:[b,seq_len,hidden_dim] -> q, k, v: [b, seq_len, hidden_dim] -> multi_head qkv:[b, seq_len, n_head, att_size]
        batch_size = x.size(0) # number of sequences to train a time
        k,v= None,None
        compute_cache_k,compute_cache_v = None,None
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.att_size)
        dup_nodes_kv_cache = dup_nodes_kv_cache[layer]
        if dup_nodes_kv_cache is None:
            k = compute_cache_k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.att_size)
            v = compute_cache_v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.att_size)
        else:
            # dup_nodes_kv_cache -> [n_layer,2,dup_nodes_num,feature_dim]
            # TODO:use kv cache
            layer_cache = dup_nodes_kv_cache[layer]  # [2, dup_nodes_num, feature_dim]
            k_cache, v_cache = layer_cache  # each: [dup_nodes_num, feature_dim]
            dup_nodes_num = k_cache.size(0)
            assert dup_nodes_num < seq_len, f"Cache size {dup_nodes_num} > input seq_len {seq_len}"
            # 1. Compute K/V for the non-cached part (from x[:, dup_nodes_num:, :])
            x_dynamic = x[:, dup_nodes_num:, :]  # [b, dynamic_len, hidden_dim]
            k_dynamic = self.linear_k(x_dynamic).view(batch_size, -1, self.num_heads, self.att_size)
            v_dynamic = self.linear_v(x_dynamic).view(batch_size, -1, self.num_heads, self.att_size)
            # 2. Reshape cached K/V to multi-head format
            # [dup_nodes_num, feature_dim] -> [1, dup_nodes_num, num_heads, att_size]
            k_cached = k_cache.view(1, dup_nodes_num, self.num_heads, self.att_size).expand(batch_size, -1, -1, -1)
            v_cached = v_cache.view(1, dup_nodes_num, self.num_heads, self.att_size).expand(batch_size, -1, -1, -1)
            # 3. Concatenate cached and dynamic parts along sequence dimension
            k = torch.cat([k_cached, k_dynamic], dim=1)  # [b, seq_len, n_head, att_size]
            v = torch.cat([v_cached, v_dynamic], dim=1)
        # print(f'rank {get_sequence_parallel_rank()} q: {q[:, 0, :, :]}')
        # exit(0)

        # ==================================
        # core attention computation
        # ==================================
        log(f"q:{q.shape},k:{k.shape},v:{v.shape}")
        # ======================================= 取消跨设备分head操作，每个设备独立计算自己的窗口。
        x,score = self.local_attn(q, k, v, attn_bias, edge_index, attn_type, mask=mask, pruning_mask=pruning_mask)
        # x,score = self.dist_attn(q, k, v, attn_bias, edge_index, attn_type,pruning_mask=pruning_mask,mask=mask)

        # =================
        # linear
        # =================

        # [b, seq_len, h]
        log(f"x shape:{x.shape}")
        assert x.size() == orig_q_size
        return x,score,[compute_cache_k,compute_cache_v]# 对分数的绝对值求和，考虑正负数都起作用


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads
        )
        
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.O = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
            
        # FFN
        self.FFN_layer1 = nn.Linear(hidden_size, hidden_size*2)
        self.FFN_layer2 = nn.Linear(hidden_size*2, hidden_size)

        self.layer_norm2 = nn.LayerNorm(hidden_size)
            
            
    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None,mask= None,pruning_mask=None,dup_nodes_kv_cache=None,layer:int=0):
        # ==================================
        # MHA
        # ==================================
        # x: [b, seq_len, hidden_dim]
        y,score,[compute_cache_k,compute_cache_v] = self.self_attention(x, attn_bias, edge_index=edge_index, 
                                      attn_type=attn_type,pruning_mask=pruning_mask,mask=mask,duplicated_nodes=dup_nodes_kv_cache,layer=layer
        )
        y = self.self_attention_dropout(y)
        y = self.O(y)
        x = x + y
        x = self.layer_norm1(x)

        # ==================================
        # MLP
        # ==================================

        y = self.FFN_layer1(y)                  # 原代码这里的输入是 y ??? 
        # y = self.FFN_layer1(x)
        y = F.relu(y)
        y = self.self_attention_dropout(y)
        y = self.FFN_layer2(y)
        x = x + y
        x = self.layer_norm2(x)

        return x,score,[compute_cache_k,compute_cache_v]
        
        
class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y



## =================================Centrality Encoding====================================

class CentralityEncodingLayer(nn.Module):
    def __init__(self,hidden_dim,num_in_degree=512, num_out_degree=512):
        """
            num_in_degree:节点最大入度，513，0代表填充位
            num_out_degree: 节点最大出度
            hidden_dim:隐层维度
        """
        super().__init__()
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.in_degree_encoder = nn.Embedding(num_in_degree + 1, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree + 1, hidden_dim, padding_idx=0)

    def forward(self,x,in_degree,out_degree):
        
        """
            x         : Tensor , 每个节点的 feature
            in_degree : Tensor , 每个节点的入度
            out_degree: Tensor , 每个节点的出度
        """
        
        in_degree_embedding = self.in_degree_encoder(in_degree.long())
        out_degree_embedding = self.out_degree_encoder(out_degree.long())
        
        # 截断,把出入度大于 512 的点统一归类为 512
        # 为什么 -1？因为索引从0开始，最大索引是 num_in_degree - 1
        # 比如 size=512，最大能接受的索引是 511
        in_degree = in_degree.clamp(max=self.num_in_degree - 1)
        out_degree = out_degree.clamp(max=self.num_out_degree - 1)
        
        x = x + in_degree_embedding + out_degree_embedding
        return x       
## ========================================================================================


## ================================== AttnBias =====================================
class AttnBias(nn.Module):
    def __init__(
        self, 
        num_heads, 
        num_spatial=512, 
        num_edges=1024, 
        max_dist=32,             # 最短路径的最大长度 即路径上最多有多少个edge
        edge_dim=32,             # 文档中的 d_E (边特征维度)
    ):
        """
        输出形状: [Batch, Heads, N, N]
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        self.edge_dim = edge_dim

        # ================== 1. Spatial Encoding  ==================
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        # ================== 2. Edge Encoding  ==================
        self.edge_feature_encoder = nn.Embedding(num_edges + 1, edge_dim, padding_idx=0)    # embed evey edge as "edge_dim"

        self.edge_pos_encoder = nn.Embedding(max_dist, edge_dim * num_heads)

    def forward(self, spatial_pos, edge_input):
        """
        Args:
            spatial_pos: [Batch, N, N]
            edge_input:  [Batch, N, N, Max_Dist] 记录了 i 到 j 的路径上第 k 条边是什么类型(是第一步，第二步...第Max_Dist步?)
            
        Returns:
            local_bias:  [Batch, Heads, N, N]
        """
        
        # ================== A. 计算 Spatial Bias ==================
        spatial_bias = self.spatial_pos_encoder(spatial_pos.long())   # spatial_pos:[b, seq_len,seq_len] -> spatial_bias: [b, seq_len,seq_len,num_heads]
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)               # [b, seq_len,seq_len,num_heads] -> [b,num_heads,seq_len,seq_len]


        # # ================== B. 计算 Edge Bias ==================
        # # 1. 获取边特征 x_{en}
        # # [B, N, N, K] -> [B, N, N, K, Edge_Dim]
        # edge_feat = self.edge_feature_encoder(edge_input.long())
        
        # # 2. 获取位置权重 w_n^E
        # # 生成位置索引: 0, 1, ..., K-1
        # # [K]
        # pos_idx = torch.arange(self.max_dist, device=edge_input.device)
        # # [K, Edge_Dim * H]
        # pos_weight = self.edge_pos_encoder(pos_idx)
        # # [K, Edge_Dim, H]
        # pos_weight = pos_weight.view(self.max_dist, self.edge_dim, self.num_heads)
        
        # # 3. 计算点积 x * w^T 
        # # dimensions:
        # # b, n1, n2: Batch, N, N
        # # k: Max_Dist (路径长度)
        # # d: Edge_Dim
        # # h: Num_Heads
        # # 公式: sum_over_d (edge_feat * pos_weight) -> [B, N, N, K, H]
        # edge_bias_terms = torch.einsum("...kd,kdh->...kh", edge_feat, pos_weight)
        
        # # 4. 计算平均值 (1/N * sum)
        # # 我们需要处理 padding 的部分 (padding_idx=0 的边不应该计入分母)
        # # 创建 mask: [B, N, N, K]
        # mask = (edge_input != 0).float()
        # # 路径长度 N: [B, N, N, 1]
        # path_len = mask.sum(dim=-1, keepdim=True)
        # # 避免除以 0
        # path_len = path_len.clamp(min=1.0)
        
        # # 求和: [B, N, N, H]
        # edge_bias_sum = edge_bias_terms.sum(dim=-2)
        
        # # 平均: [B, N, N, H]
        # edge_bias = edge_bias_sum / path_len
        
        # # 调整维度 -> [B, H, N, N]
        # edge_bias = edge_bias.permute(0, 3, 1, 2)
        
        # # ================== C. 融合 ==================
        # total_bias = spatial_bias + edge_bias
        total_bias = spatial_bias
        return total_bias
        # return None 

## ==================================================================================




class GT_SW(nn.Module):
    """GT for node-level task.
    No global token.

    """
    def __init__(
        self,
        n_layers,
        num_heads,
        input_dim,
        hidden_dim,
        output_dim,
        attn_bias_dim,
        dropout_rate,
        input_dropout_rate,
        attention_dropout_rate,
        ffn_dim,
        num_global_node,
        args,
        # ===== centrality encoding =====
        num_in_degree: int,
        num_out_degree: int,
        
        # ===== =================== =====
        num_spatial,
        num_edges,
        max_dist,
        edge_dim
        
    ):
        super().__init__()
        self.args = args
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        
        # ===== centrality encoding =====
        self.centrality_encoding = CentralityEncodingLayer(hidden_dim=hidden_dim, num_in_degree = num_in_degree,num_out_degree = num_out_degree)
        # =====                     =====
 
        # ===== attn_bias =====
        self.attention_bias = AttnBias(
            num_heads=num_heads,
            num_spatial=num_spatial,
            num_edges=num_edges,
            max_dist=max_dist,
            edge_dim=edge_dim # 可以调这个参数，越大边特征越丰富
        )
        # ===== ========= =====
        
 
        self.input_dropout = nn.Dropout(input_dropout_rate)
        
        encoders = [
            EncoderLayer(
                hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)

        self.MLP_layer = MLPReadout(hidden_dim, output_dim)   # 1 out dim since regression problem  
        self.downstream_out_proj = nn.Linear(hidden_dim, output_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        
        
    def forward(self, x, attn_bias, edge_index, in_degree, out_degree,spatial_pos, 
                edge_input,perturb=None, attn_type=None, mask= None, pruning_mask=None,dup_nodes_kv_cache=None,part_id:int=None
        ):
        if self.args.use_cache == 1:
            assert dup_nodes_kv_cache is not None
            assert dup_nodes_kv_cache[0].shape == (self.args.n_layers,2,x.shape[0],self.args.hidden_dim,self.args.hidden_dim)
        # dup_nodes_kv_cache -> [partition_id,n_layer,dup_nodes_num,2,feature_dim]
        # x -> [bs=1, s/p, x_d]
        x = x.unsqueeze(0) 
        n_graph = x.shape[0] 
        
        # [bs, s/p, x_d] -> [bs, s/p, h]
        # t1 = time.time()
        node_feature = self.node_encoder(x)
        # t2 = time.time()
        # print(f"cost time2:{t2-t1:.3f}")
        # ===== centrality encoding =====
        if self.args.struct_enc=="True":
            in_degree = in_degree.unsqueeze(0) if in_degree is not None else None
            out_degree = out_degree.unsqueeze(0) if out_degree is not None else None
            node_feature = self.centrality_encoding(node_feature, in_degree, out_degree)
        # =====                     =====
        # t3 = time.time()
        # print(f"cost time3:{t3-t2:.3f}")
        # =====   attention_bias    =====
        if self.args.struct_enc=="True":
            spatial_pos = spatial_pos.unsqueeze(0) if spatial_pos is not None else None
            edge_input = edge_input.unsqueeze(0) if edge_input is not None else None
            bias = self.attention_bias(spatial_pos, edge_input)
            if attn_bias is not None:
                attn_bias = bias + attn_bias
            else:
                attn_bias = bias
        # =====                     =====     
        # t4 = time.time()
        # print(f"cost time4:{t4-t3:.3f}")
        output = self.input_dropout(node_feature)
        # t5 = time.time()
        # print(f"cost time5:{t5-t4:.3f}")
        # [b, s/p+1, h]
        score_agg = None
        score_spe = []
        kv_cache = []
        for i,enc_layer in enumerate(self.layers):
            log(f"output shape:{output.shape}")
            output,score,[compute_cache_k,compute_cache_v] = enc_layer(
                output,
                attn_bias = attn_bias,
                edge_index=edge_index,
                attn_type=attn_type,
                mask=mask,
                dup_nodes_kv_cache=dup_nodes_kv_cache[part_id],
                layer=i
            )
            # score_agg = score if score_agg==None else score_agg+torch.sum(score,dim=1).squeeze(0) # 返回的score已经是绝对值了
            score_spe.append(score)
        # Output part
        # t6 = time.time()
        # print(f"cost time6:{t6-t5:.3f}")
        
        log(f"final output:{output.shape}")
        output = self.MLP_layer(output[0, :, :])
        log(f"final output:{output.shape}")
        
        return F.log_softmax(output, dim=1),score_agg,score_spe,dup_nodes_kv_cache