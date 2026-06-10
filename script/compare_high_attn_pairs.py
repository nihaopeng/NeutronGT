#!/usr/bin/env python3
"""
对比 PPR 分区 vs 随机 minibatch 对高分注意力对的捕获能力。

方法:
  1. 用 full-attention（标准随机 minibatch）训练 GT_SW 模型
  2. 训练完后对所有节点做一次全图 forward，得到全局 [N, N] 注意力分数矩阵
  3. 取分数的第 P 百分位作为阈值，高于此阈值的 pair 视为"高分注意力对"
  4. 统计 PPR 分区捕获了多少高分对 → PPR capture ratio
  5. 对比随机等大小分组能捕获多少 → Random expected capture ratio

用法:
    python script/compare_high_attn_pairs.py \
        --dataset_dir ./dataset/ --dataset cora \
        --n_parts 20 --percentile 90 \
        --epochs 1000 --batch_size 512 \
        --n_layers 4 --hidden_dim 64 --num_heads 8 --ffn_dim 64 \
        --device 0
"""

import argparse
import os
import sys
import random as rand
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import subgraph

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.node_level_pipeline.preprocess_cache import (
    load_preprocess_cache,
)
from core.node_level_pipeline.graph_data import _ensure_edge_index, _get_node_degrees_from_csr, load_optional_edge_csr
from models.gt_dist_node_level_single_window import GT_SW
from gt_sp.utils import get_node_degrees, random_split_idx
from utils.lr import PolynomialDecayLR

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
# Args
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='PPR vs Random: high-attention pair capture analysis')
    # Data
    p.add_argument('--dataset_dir', type=str, default='./dataset/')
    p.add_argument('--dataset', type=str, default='cora')
    # PPR
    p.add_argument('--ppr_backend', type=str, default='torch_geometric', choices=['torch_geometric', 'appnp'])
    p.add_argument('--ppr_topk', type=int, default=5)
    p.add_argument('--ppr_alpha', type=float, default=0.85)
    p.add_argument('--ppr_num_iterations', type=int, default=10)
    p.add_argument('--ppr_batch_size', type=int, default=8)
    p.add_argument('--ppr_iter_topk', type=int, default=0)
    p.add_argument('--ppr_eps', type=float, default=1e-6)
    p.add_argument('--n_parts', type=int, default=20)
    p.add_argument('--related_nodes_topk_rate', type=int, default=2)
    # Model
    p.add_argument('--model', type=str, default='gt_sw')
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--ffn_dim', type=int, default=64)
    p.add_argument('--attn_bias_dim', type=int, default=1)
    p.add_argument('--dropout_rate', type=float, default=0.3)
    p.add_argument('--input_dropout_rate', type=float, default=0.1)
    p.add_argument('--attention_dropout_rate', type=float, default=0.5)
    p.add_argument('--num_global_node', type=int, default=1)
    p.add_argument('--attn_type', type=str, default='full')
    p.add_argument('--max_dist', type=int, default=5)
    p.add_argument('--struct_enc', type=str, default='True')
    p.add_argument('--max_num_edges', type=int, default=512)
    p.add_argument('--edge_dim', type=int, default=64)
    # Training
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=512, help='Minibatch size for full-attention training')
    p.add_argument('--peak_lr', type=float, default=1e-4)
    p.add_argument('--end_lr', type=float, default=1e-9)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_updates', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    # Checkpoint (optional, skips training if provided)
    p.add_argument('--model_ckpt', type=str, default='', help='Skip training and load this checkpoint')
    # Analysis
    p.add_argument('--percentile', type=float, default=90.0, help='Percentile for high-attn threshold')
    p.add_argument('--percentiles', type=str, default='', help='Comma-separated percentiles, e.g. "75,90,95,99"')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output_dir', type=str, default='./exps/attn_pair_compare/')
    # Cache
    p.add_argument('--use_preprocess_cache', type=int, default=1, choices=[0, 1])
    p.add_argument('--refresh_preprocess_cache', type=int, default=0, choices=[0, 1])
    p.add_argument('--world_size', type=int, default=1)
    return p.parse_args()


class MockArgs:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ============================================================
# Data loading
# ============================================================
def load_dataset(args):
    dp = os.path.join(args.dataset_dir, args.dataset)
    feature = torch.load(os.path.join(dp, 'x.pt'), map_location='cpu')
    y = torch.load(os.path.join(dp, 'y.pt'), map_location='cpu')
    ecsr = load_optional_edge_csr(args.dataset_dir, args.dataset)
    ei = None if ecsr is not None else torch.load(os.path.join(dp, 'edge_index.pt'), map_location='cpu')
    return feature, y, ei, ecsr, feature.shape[0]


def load_ppr_cache(args, feature, edge_index, edge_csr_data, N):
    """加载 PPR 分区缓存"""
    mock = MockArgs(
        dataset_dir=args.dataset_dir, dataset=args.dataset,
        ppr_backend=args.ppr_backend, ppr_alpha=args.ppr_alpha, ppr_topk=args.ppr_topk,
        ppr_num_iterations=args.ppr_num_iterations, ppr_batch_size=args.ppr_batch_size,
        ppr_iter_topk=args.ppr_iter_topk, ppr_eps=args.ppr_eps,
        n_parts=args.n_parts, related_nodes_topk_rate=args.related_nodes_topk_rate,
        attn_type=args.attn_type, struct_enc=args.struct_enc, max_dist=args.max_dist,
        use_preprocess_cache=args.use_preprocess_cache,
        refresh_preprocess_cache=args.refresh_preprocess_cache,
        rank=0,
    )
    if args.use_preprocess_cache and not args.refresh_preprocess_cache:
        payload, ck, cp, _, lt = load_preprocess_cache(
            mock, graph_in_degree=None, graph_out_degree=None,
            edge_index=edge_index, edge_csr_data=edge_csr_data, num_nodes=N,
            world_size=args.world_size,
        )
        if payload is not None:
            print(f"✅ Preprocess cache hit: {cp}")
            if edge_csr_data is not None:
                gid, god = _get_node_degrees_from_csr(edge_csr_data, N)
            else:
                gid, god = get_node_degrees(edge_index, N)
            gei = _ensure_edge_index(edge_index, edge_csr_data)
            return {
                'partitioned_results': payload['wm']['partitioned_results'],
                'graph_in_degree': gid, 'graph_out_degree': god,
                'graph_edge_index': gei,
            }

    # Fallback: compute from scratch
    print("⚠️  Cache miss. Computing PPR + Metis from scratch...")
    from core.ppr_preprocess import personal_pagerank, build_adj_fromat, add_isolated_connections
    from core.metisPartition import weightMetis_keepParent

    dv = f'cuda:{args.device}' if args.device >= 0 else 'cpu'
    if edge_csr_data is not None:
        gid, god = _get_node_degrees_from_csr(edge_csr_data, N)
    else:
        gid, god = get_node_degrees(edge_index, N)

    sm = personal_pagerank(edge_index, args.ppr_alpha, topk=args.ppr_topk,
                           backend=args.ppr_backend, num_iterations=args.ppr_num_iterations,
                           batch_size=args.ppr_batch_size, eps=args.ppr_eps,
                           device=dv, csr_data=edge_csr_data, num_nodes=N,
                           iter_topk=args.ppr_iter_topk)
    gei = _ensure_edge_index(edge_index, edge_csr_data)
    sm = add_isolated_connections(sm, gei, N)
    adj, eweights, _ = build_adj_fromat(sorted_ppr_matrix=sm)
    wm = weightMetis_keepParent(csr_adjacency=adj, eweights=eweights, feature=feature,
                                edge_index=gei, n_parts=args.n_parts,
                                related_nodes_topk_rate=args.related_nodes_topk_rate,
                                attn_type=args.attn_type, sorted_ppr_matrix=sm)
    return {
        'partitioned_results': wm.partitioned_results,
        'graph_in_degree': gid, 'graph_out_degree': god,
        'graph_edge_index': gei,
    }


# ============================================================
# Model helpers
# ============================================================
def build_model(args, feature, y, device):
    out_dim = int(y.max().item()) + 1
    return GT_SW(
        n_layers=args.n_layers, num_heads=args.num_heads,
        input_dim=feature.shape[1], hidden_dim=args.hidden_dim, output_dim=out_dim,
        attn_bias_dim=args.attn_bias_dim, dropout_rate=args.dropout_rate,
        input_dropout_rate=args.input_dropout_rate,
        attention_dropout_rate=args.attention_dropout_rate,
        ffn_dim=args.ffn_dim, num_global_node=args.num_global_node,
        args=args, num_in_degree=args.num_in_degree, num_out_degree=args.num_out_degree,
        num_spatial=args.max_dist + 2, num_edges=args.max_num_edges,
        max_dist=args.max_dist, edge_dim=args.edge_dim,
    ).to(device)


def load_ckpt(model, path, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd, strict=False)
    print(f"✅ Loaded checkpoint: {path}")


def patch_model(model):
    """Monkey-patch CoreAttention.full_attention 以捕获原始 QK^T 矩阵 [N, N]"""
    from models.gt_dist_node_level_single_window import CoreAttention as CA

    def patched(self, k, q, v, attn_bias, mask=None, pruning_mask=None):
        q_ = q.transpose(1, 2)
        v_ = v.transpose(1, 2)
        k_ = k.transpose(1, 2).transpose(2, 3)
        q_ = q_ * self.scale
        x = torch.matmul(q_, k_)
        raw = x.clone()
        if attn_bias is not None:
            x = x + attn_bias
        if pruning_mask is not None:
            x = x + pruning_mask
        if mask is not None:
            m = mask.to(x.device).unsqueeze(0).unsqueeze(0).repeat(1, x.shape[1], 1, 1)
            x = x.masked_fill(m, -1e9)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v_)
        x = x.transpose(1, 2).contiguous()
        ns = torch.abs(raw).mean(dim=1).squeeze(0).sum(dim=0)
        # 存原始 QK^T, abs, mean over heads → [N, N] on CPU
        self._captured = torch.abs(raw).mean(dim=1).squeeze(0).detach().cpu()
        return x, ns

    cnt = 0
    for m in model.modules():
        if isinstance(m, CA):
            m._captured = None
            m.full_attention = patched.__get__(m, CA)
            cnt += 1
    print(f"🔧 Patched {cnt} CoreAttention layers")


def collect_last_score(model):
    """收集被 patch 后模型最后一层的 [N, N] 注意力分数矩阵"""
    from models.gt_dist_node_level_single_window import CoreAttention as CA
    mats = []
    for m in model.modules():
        if isinstance(m, CA) and m._captured is not None:
            mats.append(m._captured)
    return mats[-1] if mats else torch.zeros((0, 0))


# ============================================================
# Training: full-attention minibatch
# ============================================================
def train_full_attention(args, model, feature, y, edge_index, N, train_idx, device):
    """标准随机 minibatch 训练（full attention within each batch）"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    scheduler = PolynomialDecayLR(optimizer, warmup=args.warmup_updates, tot=args.epochs,
                                  lr=args.peak_lr, end_lr=args.end_lr, power=1.0)

    # Pre-compute global struct data
    graph_in_degree, graph_out_degree = get_node_degrees(edge_index, N)
    print(f"  Max in-degree: {graph_in_degree.max().item()}, max out-degree: {graph_out_degree.max().item()}")

    train_nodes = train_idx.tolist()
    n_train = len(train_nodes)

    print(f"Training {args.epochs} epochs, batch_size={args.batch_size}, "
          f"train_nodes={n_train}, batches/epoch≈{n_train // args.batch_size + 1}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        rand.shuffle(train_nodes)

        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            idx_i = torch.tensor(train_nodes[start:end], dtype=torch.long)
            Ni = idx_i.numel()

            x_i = feature[idx_i].to(device)
            y_i = y[idx_i].to(device)

            # Build subgraph edge_index
            ei_i, _ = subgraph(idx_i, edge_index, relabel_nodes=True, num_nodes=N)
            ei_i = ei_i.to(device)

            in_deg = graph_in_degree[idx_i].to(device)
            out_deg = graph_out_degree[idx_i].to(device)

            # spatial_pos
            sp_i = None
            if args.struct_enc == 'True':
                dist = torch.full((Ni, Ni), args.max_dist + 1, dtype=torch.long)
                for d in range(Ni):
                    dist[d, d] = 0
                adj = [[] for _ in range(Ni)]
                if ei_i.numel() > 0:
                    for u, v in zip(ei_i[0].tolist(), ei_i[1].tolist()):
                        adj[u].append(v)
                for src in range(Ni):
                    q = deque([src])
                    while q:
                        u = q.popleft()
                        cd = int(dist[src, u].item())
                        if cd >= args.max_dist:
                            continue
                        for v in adj[u]:
                            if dist[src, v] > cd + 1:
                                dist[src, v] = cd + 1
                                q.append(v)
                sp_i = torch.zeros_like(dist)
                r = dist <= args.max_dist
                sp_i[r] = dist[r] + 1
                sp_i = sp_i.to(device)

            out_i, _, _, _ = model(
                x_i, attn_bias=None, edge_index=ei_i,
                in_degree=in_deg, out_degree=out_deg,
                spatial_pos=sp_i, edge_input=None,
                attn_type=args.attn_type, mask=None,
            )
            loss = F.nll_loss(out_i, y_i.long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 100 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  Epoch {epoch:4d}/{args.epochs} | loss={total_loss / n_batches:.4f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    print("Training complete.")
    return model


# ============================================================
# Global attention matrix computation
# ============================================================
@torch.inference_mode()
def compute_global_attention_matrix(args, model, feature, edge_index, N, partition_cache, device):
    """
    对所有 N 个节点做一次 full forward，用 patched model 捕获最后一层的
    全局 [N, N] 原始 QK^T 注意力分数矩阵（abs, mean over heads）。
    对于大图，分 chunk 计算来节省显存。
    """
    graph_in_degree = partition_cache['graph_in_degree']
    graph_out_degree = partition_cache['graph_out_degree']

    # 判断是否需要分 chunk
    max_nodes_per_forward = 5000  # heuristic
    if N <= max_nodes_per_forward:
        print(f"  Computing global attention matrix: {N} nodes in single forward...")
        all_idx = torch.arange(N, dtype=torch.long)
        x_all = feature.to(device)
        ei_all = edge_index.to(device)
        in_deg = graph_in_degree.to(device)
        out_deg = graph_out_degree.to(device)

        sp_all = None
        if args.struct_enc == 'True':
            dist = torch.full((N, N), args.max_dist + 1, dtype=torch.long)
            for d in range(N):
                dist[d, d] = 0
            adj = [[] for _ in range(N)]
            if ei_all.numel() > 0:
                for u, v in zip(ei_all[0].cpu().tolist(), ei_all[1].cpu().tolist()):
                    adj[u].append(v)
            for src in tqdm(range(N), desc="spatial_pos"):
                q = deque([src])
                while q:
                    u = q.popleft()
                    cd = int(dist[src, u].item())
                    if cd >= args.max_dist:
                        continue
                    for v in adj[u]:
                        if dist[src, v] > cd + 1:
                            dist[src, v] = cd + 1
                            q.append(v)
            sp_all = torch.zeros_like(dist)
            r = dist <= args.max_dist
            sp_all[r] = dist[r] + 1
            sp_all = sp_all.to(device)

        model(x_all, attn_bias=None, edge_index=ei_all,
              in_degree=in_deg, out_degree=out_deg,
              spatial_pos=sp_all, edge_input=None,
              attn_type=args.attn_type, mask=None)
        return collect_last_score(model)  # [N, N]

    else:
        # Chunked computation — 逐 chunk 计算行块
        print(f"  Computing global attention matrix: {N} nodes in chunks of {max_nodes_per_forward}...")
        result_rows = []
        for chunk_start in tqdm(range(0, N, max_nodes_per_forward), desc="Attention chunks"):
            chunk_end = min(chunk_start + max_nodes_per_forward, N)
            chunk_idx = torch.arange(chunk_start, chunk_end, dtype=torch.long)
            chunk_N = chunk_idx.numel()

            # 构造全节点 + 此 chunk — 需要整个图作为 context 来计算对此 chunk 的 attention
            # 简化方案：只用 chunk 内的节点做 forward，但这会遗漏跨 chunk 的 pairs
            # 更准确但更慢的方案：对每个 chunk 单独 forward

            x_c = feature[chunk_idx].to(device)
            ei_c, _ = subgraph(chunk_idx, edge_index, relabel_nodes=True, num_nodes=N)
            ei_c = ei_c.to(device)
            in_c = graph_in_degree[chunk_idx].to(device)
            out_c = graph_out_degree[chunk_idx].to(device)

            sp_c = None
            if args.struct_enc == 'True':
                dist = torch.full((chunk_N, chunk_N), args.max_dist + 1, dtype=torch.long)
                for d in range(chunk_N):
                    dist[d, d] = 0
                adj = [[] for _ in range(chunk_N)]
                if ei_c.numel() > 0:
                    for u, v in zip(ei_c[0].tolist(), ei_c[1].tolist()):
                        adj[u].append(v)
                for src in range(chunk_N):
                    q = deque([src])
                    while q:
                        u = q.popleft()
                        cd = int(dist[src, u].item())
                        if cd >= args.max_dist:
                            continue
                        for v in adj[u]:
                            if dist[src, v] > cd + 1:
                                dist[src, v] = cd + 1
                                q.append(v)
                sp_c = torch.zeros_like(dist)
                r = dist <= args.max_dist
                sp_c[r] = dist[r] + 1
                sp_c = sp_c.to(device)

            model(x_c, attn_bias=None, edge_index=ei_c,
                  in_degree=in_c, out_degree=out_c,
                  spatial_pos=sp_c, edge_input=None,
                  attn_type=args.attn_type, mask=None)
            row_block = collect_last_score(model)  # [chunk_N, chunk_N]
            result_rows.append((chunk_idx, row_block))

        # 组装成完整 [N, N] 矩阵（跨 chunk 的 pair 填 0 或 -inf）
        full = torch.zeros((N, N))
        for chunk_idx, block in result_rows:
            for local_i, global_i in enumerate(chunk_idx.tolist()):
                full[global_i, chunk_idx] = block[local_i]
        return full


# ============================================================
# Capture analysis
# ============================================================
def analyze_capture(global_score_matrix, partitioned_results, percentile, N):
    """
    分析 PPR 分区对高注意力对的捕获能力。

    Args:
        global_score_matrix: [N, N] 全局注意力分数, |QK^T| mean over heads
        partitioned_results: list of Tensor, 每个 Tensor 是一个分区的全局节点索引
        percentile: float, 百分位阈值 (e.g. 90)
        N: 总节点数

    Returns:
        dict with ppr_capture_ratio, random_expected_ratio, threshold, n_high_pairs, etc.
    """
    # 取非对角元素
    off_mask = ~torch.eye(N, dtype=torch.bool)
    off_scores = global_score_matrix[off_mask]
    total_pairs = off_scores.numel()

    # 全局阈值
    thr = off_scores.quantile(percentile / 100.0).item()

    # 高分对集合
    high_mask = global_score_matrix > thr
    high_mask.fill_diagonal_(False)
    n_high_pairs = int(high_mask.sum().item())

    if n_high_pairs == 0:
        return {'error': f'No pairs exceed threshold {thr:.6f}'}

    # PPR capture: 每个分区内统计高分对个数
    ppr_captured = 0
    for part in partitioned_results:
        nodes = part.tolist()
        if len(nodes) <= 1:
            continue
        # 取子矩阵
        idx = torch.tensor(nodes, dtype=torch.long)
        sub = high_mask[idx][:, idx]
        ppr_captured += int(sub.sum().item())

    ppr_capture_ratio = ppr_captured / n_high_pairs

    # Random expected: Σ n_i*(n_i-1) / (N*(N-1))
    part_sizes = [int(p.numel()) for p in partitioned_results]
    expected_random_ratio = sum(ni * (ni - 1) for ni in part_sizes) / (N * (N - 1))

    # 也模拟几次随机分组来获得 variance
    simulated_ratios = []
    n_sim = 30
    all_nodes_list = list(range(N))
    for _ in range(n_sim):
        rand.shuffle(all_nodes_list)
        captured = 0
        offset = 0
        for ni in part_sizes:
            if ni <= 1:
                continue
            batch_nodes = all_nodes_list[offset:offset + ni]
            offset += ni
            idx = torch.tensor(batch_nodes, dtype=torch.long)
            sub = high_mask[idx][:, idx]
            captured += int(sub.sum().item())
        simulated_ratios.append(captured / n_high_pairs)

    return {
        'threshold': thr,
        'n_high_pairs': n_high_pairs,
        'total_pairs': total_pairs,
        'ppr_capture_ratio': ppr_capture_ratio,
        'random_expected_ratio': expected_random_ratio,
        'random_simulated_ratios': simulated_ratios,
        'random_simulated_mean': np.mean(simulated_ratios),
        'random_simulated_std': np.std(simulated_ratios),
        'ppr_vs_random_ratio': ppr_capture_ratio / expected_random_ratio if expected_random_ratio > 0 else float('inf'),
    }


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    percentiles = ([float(x.strip()) for x in args.percentiles.split(',') if x.strip()]
                   if args.percentiles else [args.percentile])
    device = torch.device(f'cuda:{args.device}' if args.device >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Percentiles: {percentiles}")

    # Seed
    rand.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ===== 1. Load data =====
    print("\n" + "=" * 60)
    print("Step 1: Loading dataset")
    print("=" * 60)
    feature, y, edge_index, edge_csr_data, N = load_dataset(args)
    if edge_index is None:
        edge_index = _ensure_edge_index(None, edge_csr_data)
    print(f"  {args.dataset}: N={N}, feat_dim={feature.shape[1]}")

    # ===== 2. Load PPR partition cache =====
    print("\n" + "=" * 60)
    print("Step 2: Loading PPR partition cache")
    print("=" * 60)
    ppr_cache = load_ppr_cache(args, feature, edge_index, edge_csr_data, N)
    print(f"  Partitions: {len(ppr_cache['partitioned_results'])}")

    # ===== 3. Mock SP + build model =====
    import gt_sp.initialize as sp_init
    if sp_init._SEQUENCE_PARALLEL_GROUP is None:
        sp_init._SEQUENCE_PARALLEL_GROUP = object()
        sp_init._SEQUENCE_PARALLEL_WORLD_SIZE = 1
        sp_init._SEQUENCE_PARALLEL_RANK = 0

    num_in_deg = int(torch.max(ppr_cache['graph_in_degree']).item())
    num_out_deg = int(torch.max(ppr_cache['graph_out_degree']).item())
    args.num_in_degree = max(num_in_deg, args.hidden_dim)
    args.num_out_degree = max(num_out_deg, args.hidden_dim)

    model = build_model(args, feature, y, device)

    # ===== 4. Train or load checkpoint =====
    print("\n" + "=" * 60)
    print("Step 3: Model training / loading")
    print("=" * 60)
    if args.model_ckpt and os.path.exists(args.model_ckpt):
        load_ckpt(model, args.model_ckpt, device)
        model.eval()
    else:
        split_idx = random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)
        train_idx = split_idx['train']
        print(f"  Train nodes: {train_idx.numel()}")
        model = train_full_attention(args, model, feature, y, edge_index, N, train_idx, device)

    # ===== 5. Patch + compute global attention matrix =====
    print("\n" + "=" * 60)
    print("Step 4: Computing global attention matrix")
    print("=" * 60)
    patch_model(model)
    global_scores = compute_global_attention_matrix(args, model, feature, edge_index, N, ppr_cache, device)
    print(f"  Global matrix shape: {global_scores.shape}")
    off_mask = ~torch.eye(N, dtype=torch.bool)
    off_scores = global_scores[off_mask]
    print(f"  Off-diagonal scores: min={off_scores.min().item():.6f}, "
          f"max={off_scores.max().item():.6f}, "
          f"mean={off_scores.mean().item():.6f}")

    # ===== 6. Capture analysis =====
    print("\n" + "=" * 60)
    print("Step 5: Capture analysis")
    print("=" * 60)
    partitioned_results = ppr_cache['partitioned_results']
    part_sizes = [int(p.numel()) for p in partitioned_results]
    print(f"  PPR partitions: {len(partitioned_results)}, "
          f"min_size={min(part_sizes)}, max_size={max(part_sizes)}, "
          f"avg_size={np.mean(part_sizes):.1f}")

    for pct in percentiles:
        result = analyze_capture(global_scores, partitioned_results, pct, N)
        print(f"\n  --- Percentile {pct:.0f}% ---")
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
            continue
        print(f"  Global threshold (P{pct:.0f}):          {result['threshold']:.6f}")
        print(f"  Total high-attention pairs:             {result['n_high_pairs']:,} / {result['total_pairs']:,}")
        print(f"  PPR capture ratio:                      {result['ppr_capture_ratio']:.4f}  "
              f"({result['ppr_capture_ratio']*100:.2f}%)")
        print(f"  Random expected capture ratio:          {result['random_expected_ratio']:.4f}  "
              f"({result['random_expected_ratio']*100:.2f}%)")
        print(f"  Random simulated (n=30):                "
              f"mean={result['random_simulated_mean']:.4f}, std={result['random_simulated_std']:.4f}")
        print(f"  PPR / Random ratio:                     {result['ppr_vs_random_ratio']:.4f}")
        if result['ppr_vs_random_ratio'] > 1.0:
            print(f"  ✅ PPR captures {result['ppr_vs_random_ratio']:.2f}x more high-attn pairs than random!")
        else:
            print(f"  ❌ PPR captures FEWER high-attn pairs than random.")

    # ===== 7. Save =====
    os.makedirs(args.output_dir, exist_ok=True)
    sp = os.path.join(args.output_dir, f'{args.dataset}_capture_analysis.npz')
    np.savez(sp, percentiles=np.array(percentiles),
             global_thresholds=np.array([analyze_capture(global_scores, partitioned_results, p, N).get('threshold', 0)
                                         for p in percentiles]),
             ppr_capture_ratios=np.array([analyze_capture(global_scores, partitioned_results, p, N).get('ppr_capture_ratio', 0)
                                          for p in percentiles]),
             random_expected_ratios=np.array([analyze_capture(global_scores, partitioned_results, p, N).get('random_expected_ratio', 0)
                                              for p in percentiles]))
    print(f"\n✅ Saved: {sp}")

    # ===== 8. Plot =====
    if HAS_MPL:
        print("\nGenerating plots...")
        _plot(args, percentiles, global_scores, partitioned_results, N)

    print("\nDone!")


def _plot(args, percentiles, global_scores, partitioned_results, N):
    od = os.path.join(args.output_dir)
    os.makedirs(od, exist_ok=True)

    results = [analyze_capture(global_scores, partitioned_results, p, N) for p in percentiles]
    valid = [(p, r) for p, r in zip(percentiles, results) if 'error' not in r]

    if not valid:
        return

    pcts = [v[0] for v in valid]
    res = [v[1] for v in valid]

    # Bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(pcts))
    w = 0.35
    ppr_vals = [r['ppr_capture_ratio'] for r in res]
    rand_vals = [r['random_simulated_mean'] for r in res]
    rand_errs = [r['random_simulated_std'] for r in res]

    ax1.bar(x - w/2, ppr_vals, w, label='PPR capture', color='#e74c3c', alpha=0.8)
    ax1.bar(x + w/2, rand_vals, w, yerr=rand_errs, label='Random capture', color='#3498db', alpha=0.8, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'P{p:.0f}' for p in pcts])
    ax1.set_ylabel('Capture Ratio')
    ax1.set_title(f'High-Attention Pair Capture: PPR vs Random\n({args.dataset})')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ratios = [r['ppr_vs_random_ratio'] for r in res]
    ax2.bar(range(len(pcts)), ratios, color='#2ecc71', alpha=0.8)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal (=1)')
    ax2.set_xticks(range(len(pcts)))
    ax2.set_xticklabels([f'P{p:.0f}' for p in pcts])
    ax2.set_ylabel('PPR / Random Capture Ratio')
    ax2.set_title(f'PPR Advantage over Random\n({args.dataset})')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(od, f'{args.dataset}_capture_comparison.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fp}")


if __name__ == '__main__':
    main()
