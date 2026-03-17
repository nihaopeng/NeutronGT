import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import degree
from gt_sp.layer import DistributedAttentionLocalBias, DistributedAttentionAll2all, DistributedAttentionAll2allNoMerge
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
)
from torch_scatter import scatter
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        

class GraphNodeFeature(nn.Module):
    def __init__(
        self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms

        self.atom_encoder = nn.Embedding(num_atoms, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)


    def forward(self, x, in_degree, out_degree):
        n_graph, n_node = x.size()[:2]

        node_feature = self.atom_encoder(x).sum(dim=-2)
        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )
        
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphAttnBias(nn.Module):
    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_in_degree,
        num_out_degree,
        hidden_dim,
        max_dist=20,
        edge_type="undirected",
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist

        self.edge_encoder = nn.Embedding(num_edges * num_heads, 1)
        self.graph_attn_bias = nn.Embedding(num_heads, 1, padding_idx=0)
        self.spatial_encoder = nn.Embedding(num_spatial, num_heads)
        self.in_degree_encoder = nn.Embedding(num_in_degree, num_heads, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, num_heads, padding_idx=0)

    def forward(self, batch):
        pass


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_size)
        self.linear2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class CoreAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(CoreAttention, self).__init__()

        seq_parallel_world_size = 1
        if sequence_parallel_is_initialized():
            seq_parallel_world_size = get_sequence_parallel_world_size()
        world_size = seq_parallel_world_size 

        self.hidden_size_per_partition = hidden_size // world_size
        self.hidden_size_per_attention_head = hidden_size // num_heads
        self.num_attention_heads_per_partition = num_heads // world_size

        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.num_heads = num_heads
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.attention_dropout_rate = attention_dropout_rate


    def full_attention(self, k, q, v, attn_bias, mask=None):
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill(mask, 0)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        return x


    def forward(self, q, k, v, attn_bias=None, edge_index=None, attn_type=None, mask=None, pruning_mask=None):
        if attn_type == "flash":
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            k = k.transpose(1, 2)
            qkv = torch.stack([q, k, v], dim=2)
            x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attention_dropout_rate if self.training else 0.0)
            x = x.transpose(1, 2)
        elif attn_type == "full":
            x = self.full_attention(k, q, v, attn_bias, mask)
        elif attn_type == "sparse":
            raise NotImplementedError("Sparse attention not implemented for KV cache model")
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)

        local_attn = CoreAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.dist_attn = DistributedAttentionAll2allNoMerge(local_attn, get_sequence_parallel_group())

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)


    def forward(self, x, attn_bias=None, mask=None, edge_index=None, attn_type=None,
               dup_nodes_kv_cache=None, layer=0):
        orig_q_size = x.size()
        batch_size = x.size(0)
        
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.att_size)
        
        compute_cache_k, compute_cache_v = None, None
        
        if dup_nodes_kv_cache is None:
            k = compute_cache_k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.att_size)
            v = compute_cache_v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.att_size)
        else:
            layer_cache = dup_nodes_kv_cache[layer]
            k_cache, v_cache = layer_cache
            dup_nodes_num = k_cache.size(0)
            
            x_dynamic = x[:, dup_nodes_num:, :]
            k_dynamic = self.linear_k(x_dynamic).view(batch_size, -1, self.num_heads, self.att_size)
            v_dynamic = self.linear_v(x_dynamic).view(batch_size, -1, self.num_heads, self.att_size)
            
            k_cached = k_cache.unsqueeze(0).expand(batch_size, -1, -1, -1)
            v_cached = v_cache.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            k = torch.cat([k_cached, k_dynamic], dim=1)
            v = torch.cat([v_cached, v_dynamic], dim=1)

        x = self.dist_attn(q, k, v, attn_bias, edge_index, attn_type)

        x = self.output_layer(x)  

        assert x.size() == orig_q_size
        return x, [compute_cache_k, compute_cache_v]


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

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None, mask=None, edge_index=None, attn_type=None,
                dup_nodes_kv_cache=None, layer=0):
        h = x
        x = self.self_attention_norm(x)
        x, kv_cache = self.self_attention(
            x, 
            attn_bias=attn_bias, 
            mask=mask, 
            edge_index=edge_index, 
            attn_type=attn_type,
            dup_nodes_kv_cache=dup_nodes_kv_cache,
            layer=layer
        )
        x = self.self_attention_dropout(x)
        x = h + x

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = h + x

        return x, kv_cache


class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x


class Graphormer(nn.Module):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate=0.0,
        intput_dropout_rate=0.0,
        ffn_dim=0.,
        dataset_name="MalNetTiny",
        edge_type="undirected",
        multi_hop_max_dist=20,
        attention_dropout_rate=0.0,
        output_dim=1,
        args=None,
    ):
        super(Graphormer, self).__init__()
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self.dataset_name = dataset_name
        if dataset_name in ["MalNetTiny", "MalNet"]:
            num_atoms = 150
            num_edges = 150
            num_spatial = 512
            num_in_degree = 512
            num_out_degree = 512
        else:
            num_atoms = 7000
            num_edges = 7000
            num_spatial = 512
            num_in_degree = 7000
            num_out_degree = 7000
            
        self.graph_node_feature = GraphNodeFeature(
            num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim
        )
        
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        
        encoders = [
            EncoderLayer(
                hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)

        self.MLP_layer = MLPReadout(hidden_dim, output_dim)   
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        
        
    def forward(self, x, in_degree_i, out_degree_i, edge_index, attn_type=None,
                dup_nodes_kv_cache=None, part_id=None):
        x = self.graph_node_feature(x, in_degree_i, out_degree_i)
        n_graph, n_node = x.size()[:2]
        output = self.input_dropout(x)
        
        all_kv_cache = []
        
        for enc_layer in self.layers:
            output, kv_cache = enc_layer(
                output, 
                edge_index=edge_index,
                attn_type=attn_type,
                dup_nodes_kv_cache=dup_nodes_kv_cache,
            )
            if kv_cache[0] is not None:
                all_kv_cache.append((kv_cache[0].detach(), kv_cache[1].detach()))
            else:
                all_kv_cache.append((None, None))
            
        output = output[:, 0, :]
        
        output = self.MLP_layer(output) 
        
        return output, all_kv_cache
