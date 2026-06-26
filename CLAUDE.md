# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeutronGT is a window-level Graph Transformer framework for node-level graph learning. It partitions large graphs into subgraphs ("windows") using Personalized PageRank (PPR) affinity and Metis partitioning, then trains a Graph Transformer model where **each GPU independently processes its assigned windows** — attention is computed only within each window, not across windows.

**Key difference from Baseline:** The Baseline (TorchGT) uses sequence parallelism, splitting a full-graph attention computation across multiple GPUs via all-to-all communication. NeutronGT **does not use sequence parallelism** — each GPU owns its windows and computes attention locally within each window. This eliminates inter-GPU communication during attention computation and makes large-graph training feasible via sparse attention within bounded-size windows.

The `NeutronGT/` directory contains the PPR-based windowed approach. The `Baseline/` directory contains the original TorchGT approach (sequence-parallel full-graph attention, no PPR windows) used for comparison experiments.

## Environment

- **CUDA 12.1 required** for the APPNP cuSPARSE backend (`cusparse_spgemm.cu` is compiled JIT)
- Python dependencies: PyTorch, PyG, `pymetis`, `torch_scatter`, `ogb`, `dgl` (Baseline only)
- Conda environment: `gt` (used in scripts)

```shell
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
```

## Dataset Format

Datasets are stored under `./dataset/<name>/` with files:
- `x.pt` — node features
- `y.pt` — labels
- `edge_index.pt` — COO edge index `[2, E]`
- `edge_index_csr.pt` — (NeutronGT only) CSR format with `{"rowptr": ..., "col": ...}`

Run `python utils/preprocess_data.py <dataset_name>` to download and preprocess datasets from PyG/OGB.

## Key Scripts

### NeutronGT training

```shell
cd NeutronGT

# Main training (full config with sparse attention + cache)
bash scripts/run_NeutronGT.sh <gpus> --arxiv --GT        # GT model
bash scripts/run_NeutronGT.sh <gpus> --arxiv --GPH_Slim   # Graphormer slim
bash scripts/run_NeutronGT.sh <gpus> --arxiv --GPH_Large  # Graphormer large

# Ablation: full attention within windows, no KV cache (HAW only, no RWP)
bash scripts/run_ablation_2.sh <gpus> --arxiv --GPH_Slim

# Ablation: sparse attention + KV cache (HAW + RWP, i.e., full NeutronGT)
bash scripts/run_ablation_3.sh <gpus> --arxiv --GPH_Slim

# Runtime breakdown (500 epoch, GPH_Large)
bash scripts/run_Runtimebreakdown.sh <gpus> --arxiv --GPH_Large
```

Supported datasets: `--arxiv`, `--amazon`, `--reddit`, `--products`

### Baseline (TorchGT) training

```shell
cd Baseline
bash scripts/run_torchGT.sh <gpus> --arxiv --GT
bash scripts/run_ablation_1.sh <gpus> --arxiv --GPH_Slim  # full attention, seq_len=16K
```

### Direct torchrun invocation

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=<port> \
  main_sp_node_level_ppr.py \
  --dataset ogbn-arxiv --dataset_dir ./dataset/ \
  --model graphormer --attn_type sparse \
  --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 \
  --epochs 500 --use_cache 1 --use_preprocess_cache 0 \
  --n_parts 16 --related_nodes_topk_rate 8 \
  --ppr_backend appnp --ppr_topk 5 --ppr_alpha 0.85 \
  --ppr_num_iterations 10 --ppr_batch_size 8192 --ppr_iter_topk 5 \
  --distributed-backend nccl --distributed-timeout-minutes 120
```

## Architecture

### Entry point and pipeline

`NeutronGT/main_sp_node_level_ppr.py` — the single entry point. The `main()` function executes this pipeline:

1. **Distributed init** — `initialize_distributed(args)` sets up distributed process groups (used for data-parallel window distribution and gradient all-reduce; NOT sequence parallel)
2. **Data loading** — loads `x.pt`, `y.pt`, `edge_index.pt` (or `edge_index_csr.pt` for APPNP)
3. **Graph preprocessing** — `build_graph_struct_info()` computes PPR → builds Metis partitions → creates `StructInfo`
4. **Window state broadcast** — `broadcast_window_state()` distributes window assignments across ranks (each rank receives a disjoint subset of windows)
5. **Model build** — `build_model()` creates Graphormer / GT_SW
6. **Training loop** — `train_epoch()` iterates over this rank's windows independently; every 20 epochs runs `eval_epoch()`

### Core preprocessing pipeline (`core/`)

**PPR computation** (`core/ppr_preprocess.py`):
- `personal_pagerank()` dispatches to one of two backends:
  - `torch_geometric` backend — `torch_geometric.utils.ppr.get_ppr()`, CPU-friendly, for small graphs
  - `appnp` backend — cuSPARSE-based power iteration with batching, GPU-accelerated, supports distributed sharding via `source_start`/`source_end`. This is the default for large graphs.
- `build_adj_fromat()` converts PPR edge list to a pymetis-compatible CSR adjacency (symmetric, deduplicated, weights scaled to int32)
- `add_isolated_connections()` ensures no node is completely isolated in the PPR graph

**Window construction** (`core/metisPartition.py` — `weightMetis_keepParent`):
- Two-level Metis partitioning: first 2-way parent split, then each parent is further partitioned into `n_parts/2` children
- Node supplementation per window:
  - `_merge_related_nodes()` — adds external neighbors with highest edge-weight connections (halo)
  - `_merge_feature_sim()` — samples nodes from the most similar window in the other parent partition, with virtual edges
- `_find_duplicate_nodes_and_rerange()` — identifies duplicate nodes (appearing in multiple windows) and places them at the front for KV cache reuse
- `node_out()` / `node_in()` — **deprecated, abandoned feature.** Originally intended for dynamic window adjustment during training (pruning low-attention nodes, replenishing from a global expired-node buffer), but this approach is no longer pursued. The code remains in `metisPartition.py` for reference only.

**Pipeline modules** (`core/node_level_pipeline/`):
- `struct_info.py` — `StructInfo` class holds all graph/window metadata; `build_graph_struct_info()` orchestrates the full preprocessing pipeline and caches results via `preprocess_cache`
- `window_state.py` — `broadcast_window_state()` serializes per-rank window bundles to disk, each rank loads its own; `build_local_partitions()` / `build_dup_cache_metadata()` rebuild local edge indices, spatial positions, and KV cache indices
- `train_eval.py` — `train_epoch()` iterates over this rank's windows independently (no cross-GPU communication during attention); uses dummy steps for load balancing when ranks have unequal window counts. `eval_epoch()` gathers predictions across ranks for accuracy computation only on rank 0
- `graph_data.py` — utilities for loading CSR edge data and converting CSR ↔ COO
- `checkpoint.py` — resumable training checkpoints compatible with `--use_cache 1` mode only; validates checkpoint compatibility on key args
- `preprocess_cache.py` — caches the expensive PPR+Metis result keyed by hash of relevant args; avoids recomputation on reruns

### Models (`models/`)

- **`Graphormer`** (`graphormer_dist_node_level.py`) — Windowized Graphormer with CentralityEncoding (in/out degree), spatial positional encoding (AttnBias), pre-norm transformer layers. Supports KV cache for duplicate nodes. Two attention modes: `full` and `sparse`.
- **`GT_SW`** (`gt_dist_node_level_single_window.py`) — "GT Single Window" model. Same architecture as Graphormer but with an MLPReadout head instead of a single linear projection. Also handles KV cache for duplicate nodes.
- Both models receive per-window inputs: `x [N, d]`, `edge_index [2, E]`, optional `in_degree`, `out_degree`, `spatial_pos`, `dup_nodes_kv_cache`
- KV cache mechanism: duplicate nodes' K/V are precomputed once and reused across layers; the model concatenates cached K/V with dynamically computed K/V for non-duplicate nodes

### Distributed communication (`gt_sp/`)

Adapted from Microsoft DeepSpeed's sequence parallelism, but in NeutronGT only the **gradient synchronization** and **parameter broadcast** utilities are used. The attention-level `_SeqAllToAll` / `DistributedAttentionNodeLevel` is present in the code but **not used** — NeutronGT models call `self.local_attn(...)` directly (see the commented-out `# x,score = self.dist_attn(...)` in `MultiHeadAttention.forward()`).

- `initialize.py` — sets up distributed process groups, manages rank/world_size
- `gt_layer.py` — `DistributedAttentionNodeLevel` wraps local attention with all-to-all for sequence-parallel attention (used by Baseline, NOT by NeutronGT)
- `reducer.py` — `sync_params_and_buffers()` broadcasts model params across ranks

### Attention modes

- **`sparse`** — the **primary and commonly used mode.** Edge-index-based sparse attention using `torch_scatter`. Computes attention only along graph edges within each window, O(E). This is what makes large-graph training feasible.
- **`full`** — standard `Q @ K^T` dense attention, O(N²). Only practical for very small graphs or tiny windows; essentially impossible on large graphs. The ablation experiments use full attention with small `seq_len=16K` only as a controlled baseline for comparison.

### Key configuration flags

| Flag | Meaning |
|------|---------|
| `--use_cache 1` | Enable KV cache for duplicate nodes across windows (standard mode) |
| `--use_cache 0` | Disable KV cache; formerly used for abandoned dynamic window adjustment |
| `--attn_type sparse/full` | Sparse (edge-based) or full (dense) attention |
| `--use_preprocess_cache 0/1` | Cache/reuse the PPR+Metis preprocess result (only with `--use_cache 1`) |
| `--ppr_backend appnp/torch_geometric` | cuSPARSE GPU PPR vs PyG CPU PPR |
| `--n_parts` | Number of graph windows (must be even) |
| `--related_nodes_topk_rate` | % of external neighbors merged into each window |
| `--struct_enc True` | Enable centrality + spatial positional encoding |
| `--preprocess_only 1` | Stop after preprocessing, before training |

## CUDA custom op

`core/ppr_backends/csrc/cusparse_spgemm.cu` — a cuSPARSE SpGEMM kernel compiled via `torch.utils.cpp_extension.load_inline()` in `cusparse_ops.py`. First run triggers JIT compilation (slow); subsequent runs reuse the cached `.so`.
