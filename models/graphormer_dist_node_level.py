import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter

from utils.logger import log


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class CoreAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        self.hidden_size_per_attention_head = hidden_size // num_heads
        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.num_heads = num_heads
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.attention_dropout_rate = attention_dropout_rate

    def full_attention(self, k, q, v, attn_bias, mask=None, pruning_mask=None):
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q = q * self.scale
        x = torch.matmul(q, k)
        score = x

        if attn_bias is not None:
            x = x + attn_bias
        if pruning_mask is not None:
            x = x + pruning_mask
        if mask is not None:
            mask = mask.to(x.device).unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(1, x.shape[1], 1, 1)
            x = x.masked_fill(mask, -1e9)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)
        x = x.transpose(1, 2).contiguous()
        node_scores = torch.abs(score).mean(dim=1).squeeze(0).sum(dim=0)
        return x, node_scores

    def sparse_attention_bias(self, q, k, v, edge_index, attn_bias):
        batch_size, node_num = k.size(0), k.size(1)
        num_heads = self.num_heads

        q = q.view(-1, num_heads, self.hidden_size_per_attention_head)
        k = k.view(-1, num_heads, self.hidden_size_per_attention_head)
        v = v.view(-1, num_heads, self.hidden_size_per_attention_head)

        src = k[edge_index[0].to(torch.long)]
        dest = q[edge_index[1].to(torch.long)]
        score = torch.mul(src, dest)
        score = score / self.scale
        score = score.sum(-1, keepdim=True).clamp(-5, 5)

        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 2, 3, 1).contiguous().unsqueeze(2).repeat(1, 1, batch_size, 1, 1)
            attn_bias = attn_bias.view(batch_size * node_num, batch_size * node_num, num_heads)
            score = score + attn_bias[edge_index[0].to(torch.long), edge_index[1].to(torch.long), :].unsqueeze(2)

        score = torch.exp(score)

        msg = v[edge_index[0].to(torch.long)] * score
        wv = torch.zeros_like(v)
        scatter(msg, edge_index[1], dim=0, out=wv, reduce="add")

        z = score.new_zeros(v.size(0), num_heads, 1)
        scatter(score, edge_index[1], dim=0, out=z, reduce="add")

        x = wv / (z + 1e-6)
        node_scores = score.mean(dim=1).squeeze(-1)
        node_scores = scatter(node_scores, edge_index[1], dim=0, reduce="add")
        return x, node_scores

    def naive_attention(self, q, k, v, dropout_p=0.0):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=dropout_p, training=True)
        output = torch.matmul(attn_probs, v)
        return output, None

    def forward(self, q, k, v, attn_bias=None, edge_index=None, attn_type=None, mask=None, pruning_mask=None):
        batch_size, s_len = q.size(0), q.size(1)
        if attn_type == "sparse":
            x, score = self.sparse_attention_bias(q, k, v, edge_index, attn_bias)
        elif attn_type == "flash":
            q = q.half()
            k = k.half()
            v = v.half()
            x, _ = self.naive_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.attention_dropout_rate,
            )
            x = x.transpose(1, 2).contiguous().float()
            score = None
        else:
            x, score = self.full_attention(k, q, v, attn_bias, mask=mask, pruning_mask=pruning_mask)

        x = x.view(batch_size, s_len, -1)
        return x, score


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.local_attn = CoreAttention(hidden_size, attention_dropout_rate, num_heads)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)

    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None, mask=None, pruning_mask=None, dup_nodes_kv_cache=None, layer=0):
        orig_q_size = x.size()
        batch_size, seq_len, _ = x.shape

        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.att_size)
        compute_cache_k = None
        compute_cache_v = None

        if dup_nodes_kv_cache is None:
            k = compute_cache_k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.att_size)
            v = compute_cache_v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.att_size)
        else:
            layer_cache = dup_nodes_kv_cache[layer]
            k_cache, v_cache = layer_cache
            dup_nodes_num = k_cache.size(0)
            assert dup_nodes_num <= seq_len, f"Cache size {dup_nodes_num} > input seq_len {seq_len}"

            x_dynamic = x[:, dup_nodes_num:, :]
            k_dynamic = self.linear_k(x_dynamic).view(batch_size, -1, self.num_heads, self.att_size)
            v_dynamic = self.linear_v(x_dynamic).view(batch_size, -1, self.num_heads, self.att_size)

            k_cached = k_cache.unsqueeze(0).expand(batch_size, -1, -1, -1)
            v_cached = v_cache.unsqueeze(0).expand(batch_size, -1, -1, -1)

            k = torch.cat([k_cached, k_dynamic], dim=1)
            v = torch.cat([v_cached, v_dynamic], dim=1)
            compute_cache_k = k
            compute_cache_v = v

        x, score = self.local_attn(q, k, v, attn_bias, edge_index, attn_type, mask=mask, pruning_mask=pruning_mask)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x, score, [compute_cache_k, compute_cache_v]


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None, mask=None, pruning_mask=None, dup_nodes_kv_cache=None, layer=0):
        y = self.self_attention_norm(x)
        y, score, [compute_cache_k, compute_cache_v] = self.self_attention(
            y,
            attn_bias,
            edge_index,
            attn_type=attn_type,
            mask=mask,
            pruning_mask=pruning_mask,
            dup_nodes_kv_cache=dup_nodes_kv_cache,
            layer=layer,
        )
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x, score, [compute_cache_k, compute_cache_v]


class CentralityEncodingLayer(nn.Module):
    def __init__(self, hidden_dim, num_in_degree=512, num_out_degree=512):
        super().__init__()
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.in_degree_encoder = nn.Embedding(num_in_degree + 1, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree + 1, hidden_dim, padding_idx=0)

    def forward(self, x, in_degree, out_degree):
        in_degree = in_degree.clamp(max=self.num_in_degree - 1)
        out_degree = out_degree.clamp(max=self.num_out_degree - 1)
        in_degree_embedding = self.in_degree_encoder(in_degree.long())
        out_degree_embedding = self.out_degree_encoder(out_degree.long())
        x = x + in_degree_embedding + out_degree_embedding
        return x


class AttnBias(nn.Module):
    def __init__(self, num_heads, num_spatial=512, num_edges=1024, max_dist=32, edge_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        self.edge_dim = edge_dim
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.edge_feature_encoder = nn.Embedding(num_edges + 1, edge_dim, padding_idx=0)
        self.edge_pos_encoder = nn.Embedding(max_dist, edge_dim * num_heads)

    def forward(self, spatial_pos, edge_input):
        spatial_bias = self.spatial_pos_encoder(spatial_pos.long())
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)
        return spatial_bias


class Graphormer(nn.Module):
    """Windowized Graphormer for node-level task."""

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
        num_in_degree,
        num_out_degree,
        num_spatial,
        num_edges,
        max_dist,
        edge_dim,
    ):
        super().__init__()
        self.args = args
        self.num_heads = num_heads
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.centrality_encoding = CentralityEncodingLayer(
            hidden_dim=hidden_dim,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
        )
        self.attention_bias = AttnBias(
            num_heads=num_heads,
            num_spatial=num_spatial,
            num_edges=num_edges,
            max_dist=max_dist,
            edge_dim=edge_dim,
        )
        self.input_dropout = nn.Dropout(input_dropout_rate)
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.downstream_out_proj = nn.Linear(hidden_dim, output_dim)
        self.n_layers = n_layers
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(
        self,
        x,
        attn_bias,
        edge_index,
        in_degree,
        out_degree,
        spatial_pos,
        edge_input,
        perturb=None,
        attn_type=None,
        mask=None,
        pruning_mask=None,
        dup_nodes_kv_cache=None,
        part_id=None,
    ):
        x = x.unsqueeze(0)
        node_feature = self.node_encoder(x)

        if perturb is not None:
            node_feature += perturb

        if self.args.struct_enc == "True":
            in_degree = in_degree.unsqueeze(0) if in_degree is not None else None
            out_degree = out_degree.unsqueeze(0) if out_degree is not None else None
            node_feature = self.centrality_encoding(node_feature, in_degree, out_degree)

        if self.args.struct_enc == "True":
            spatial_pos = spatial_pos.unsqueeze(0) if spatial_pos is not None else None
            edge_input = edge_input.unsqueeze(0) if edge_input is not None else None
            bias = self.attention_bias(spatial_pos, edge_input)
            attn_bias = bias + attn_bias if attn_bias is not None else bias

        output = self.input_dropout(node_feature)

        score_agg = None
        score_spe = []
        new_kv_cache = []
        for i, enc_layer in enumerate(self.layers):
            layer_kv_cache = None
            if dup_nodes_kv_cache is not None:
                # dup_nodes_kv_cache already holds the current partition cache as a per-layer list.
                layer_kv_cache = dup_nodes_kv_cache

            output, score, [compute_cache_k, compute_cache_v] = enc_layer(
                output,
                attn_bias=attn_bias,
                edge_index=edge_index,
                attn_type=attn_type,
                mask=mask,
                pruning_mask=pruning_mask,
                dup_nodes_kv_cache=layer_kv_cache,
                layer=i,
            )
            score_spe.append(score.detach())

            if compute_cache_k is not None and compute_cache_v is not None:
                if dup_nodes_kv_cache is not None and layer_kv_cache is not None:
                    dup_nodes_num = 0
                    if isinstance(layer_kv_cache, tuple) and len(layer_kv_cache) == 2:
                        k_cache_item, _ = layer_kv_cache
                        if k_cache_item is not None:
                            dup_nodes_num = k_cache_item.shape[0]

                    if dup_nodes_num > 0:
                        k_cache = compute_cache_k[:, :dup_nodes_num, :, :].squeeze(0).detach()
                        v_cache = compute_cache_v[:, :dup_nodes_num, :, :].squeeze(0).detach()
                        new_kv_cache.append((k_cache, v_cache))
                    else:
                        new_kv_cache.append((None, None))
                else:
                    new_kv_cache.append((None, None))
            else:
                new_kv_cache.append((None, None))

        output = self.final_ln(output)
        log(f"final output:{output.shape}")
        output = self.downstream_out_proj(output[0, :, :])

        updated_kv_cache = dup_nodes_kv_cache
        if new_kv_cache and any(k is not None for k, _ in new_kv_cache):
            if updated_kv_cache is None:
                updated_kv_cache = []
                for k_cache, v_cache in new_kv_cache:
                    updated_kv_cache.append((k_cache, v_cache) if k_cache is not None and v_cache is not None else (None, None))
            elif isinstance(updated_kv_cache, list):
                for i, (k_cache, v_cache) in enumerate(new_kv_cache):
                    if k_cache is not None and v_cache is not None and i < len(updated_kv_cache):
                        updated_kv_cache[i] = (k_cache, v_cache)

        return F.log_softmax(output, dim=1), score_agg, score_spe, updated_kv_cache
