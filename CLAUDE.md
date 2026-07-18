# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Interaction rules

- **回答问题时只回答，不修改代码。** 只有用户明确说"改"、"修"、"commit"、"push"等指令时才动手
- **修改前先明确计划。** 说明要改哪个文件、为什么、改什么，边界清晰后再动手
- **每次改动后严格检查。** 确认改动不破坏调用链、不引入新问题

## Project Overview

NeutronGT is a window-level Graph Transformer framework for node-level graph learning. It partitions large graphs into subgraphs ("windows") using Personalized PageRank (PPR) affinity and Metis partitioning, then trains a Graph Transformer model where **each GPU independently processes its assigned windows**. Attention is computed only within each local window, not across windows.

**Key difference from Baseline:** The Baseline (TorchGT) uses sequence parallelism, splitting each sampled `seq_len` training sequence across GPUs and using all-to-all communication in the attention layer. It is not PPR-window based. NeutronGT still initializes the legacy "sequence parallel" process group, but the model attention path does **not** use sequence-parallel all-to-all; each GPU owns its assigned windows and computes attention locally within each window. This eliminates inter-GPU communication during attention computation and makes large-graph training feasible via sparse attention within bounded-size windows.

The `NeutronGT/` directory contains the PPR-based windowed approach. The `Baseline/` directory contains the original TorchGT-style approach (sequence-parallel attention over sampled/reordered sequences, no PPR windows) used for comparison experiments.

## Environment

- The provided scripts export `/usr/local/cuda-12.1`. The APPNP cuSPARSE backend itself requires a CUDA toolkit discoverable by `CUDA_HOME`/`CUDA_PATH` or `nvcc`, with `cusparse.h` available.
- Python dependencies: PyTorch, PyG, `pymetis`, `torch_scatter`, `ogb`, `dgl`; Baseline also imports `flash_attn` for flash attention mode.
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
- `edge_index_csr.pt` — optional NeutronGT CSR format with `{"rowptr": ..., "col": ...}`. APPNP uses it when present; otherwise it builds CSR from `edge_index.pt`.

Run `python utils/preprocess_data.py <dataset_name>` to download and preprocess datasets from PyG/OGB. This script saves `x.pt`, `y.pt`, and `edge_index.pt`; it does not currently generate `edge_index_csr.pt`.

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

1. **Distributed init** — `initialize_distributed(args)` sets up torch distributed and a legacy sequence-parallel group. In NeutronGT this group is used for rank/world-size helpers, parameter broadcast, gradient all-reduce, barriers, and object exchange; attention itself does not use sequence-parallel all-to-all.
2. **Data loading** — loads `x.pt` and `y.pt`; for APPNP it tries optional `edge_index_csr.pt` first, otherwise loads `edge_index.pt`
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
- `node_out()` / `node_in()` — dynamic window adjustment path. It is still reachable when `--use_cache 0` and `LossStagnationDetector` fires: rank 0 prunes low-attention nodes, refills from the expired-node buffer, increments `window_state_version`, and rebroadcasts window state. The mainstream fixed-window/cache path is `--use_cache 1`; checkpoint resume and preprocess cache only support that path.

**Pipeline modules** (`core/node_level_pipeline/`):
- `struct_info.py` — `StructInfo` class holds all graph/window metadata; `build_graph_struct_info()` orchestrates the full preprocessing pipeline and can cache results via `preprocess_cache` when `--use_cache 1 --use_preprocess_cache 1`
- `window_state.py` — `broadcast_window_state()` serializes per-rank window bundles to disk, each rank loads its own; `build_local_partitions()` / `build_dup_cache_metadata()` rebuild local edge indices, spatial positions, and KV cache indices
- `train_eval.py` — `train_epoch()` iterates over this rank's windows independently (no cross-GPU communication during attention); uses dummy steps for load balancing when ranks have unequal window counts. `eval_epoch()` gathers predictions across ranks for accuracy computation only on rank 0
- `graph_data.py` — utilities for loading CSR edge data and converting CSR ↔ COO
- `checkpoint.py` — resumable training checkpoints compatible with `--use_cache 1` mode only; validates checkpoint compatibility on key args
- `preprocess_cache.py` — caches the expensive PPR+Metis result keyed by hash of relevant args; avoids recomputation on reruns

### Models (`models/`)

- **`Graphormer`** (`graphormer_dist_node_level.py`) — Windowized Graphormer with CentralityEncoding (in/out degree), spatial positional encoding (AttnBias), pre-norm transformer layers. Supports KV cache for duplicate nodes. Two attention modes: `full` and `sparse`.
- **`GT_SW`** (`gt_dist_node_level_single_window.py`) — "GT Single Window" model. It uses the same windowized inputs and local attention idea as Graphormer, but its encoder block follows the older GT-style implementation (`O` projection, post-residual norms, ReLU FFN, MLPReadout head). Also includes KV cache handling for duplicate nodes.
- Both models receive per-window inputs: `x [N, d]`, `edge_index [2, E]`, optional `in_degree`, `out_degree`, `spatial_pos`, `dup_nodes_kv_cache`
- Intended KV cache mechanism: duplicate nodes are placed at the front of each local window, and the model can concatenate cached K/V for those duplicate nodes with dynamically computed K/V for non-duplicate nodes. Current implementation caveat: `train_epoch()` initializes each partition cache as `None`, and the model only materializes a new cache when a non-`None` cache is passed in, so the K/V reuse path appears not to warm up under the current code.

### Distributed communication (`gt_sp/`)

Adapted from Microsoft DeepSpeed's sequence parallelism, but in NeutronGT only the **distributed bookkeeping, gradient synchronization, parameter broadcast, barriers, and object/metric communication** are used. The attention-level `_SeqAllToAll` / `DistributedAttentionNodeLevel` is present in the code but **not used by NeutronGT model forward paths**; NeutronGT models call `self.local_attn(...)` directly (see the commented-out `# x,score = self.dist_attn(...)` in `GT_SW.MultiHeadAttention.forward()`). Baseline has its own copied `gt_sp/` modules and does use sequence-parallel attention.

- `initialize.py` — sets up distributed process groups, manages rank/world_size
- `gt_layer.py` — `DistributedAttentionNodeLevel` wraps local attention with all-to-all for sequence-parallel attention (used by Baseline, NOT by NeutronGT)
- `reducer.py` — `sync_params_and_buffers()` broadcasts model params across ranks

### Attention modes

- **`sparse`** — the **primary and commonly used mode.** Edge-index-based sparse attention using `torch_scatter`. Computes attention only along graph edges within each window, O(E). This is what makes large-graph training feasible.
- **`full`** — standard `Q @ K^T` dense attention, O(N^2). Only practical for very small graphs or tiny windows. In NeutronGT, `run_ablation_2.sh` uses full attention inside PPR/Metis windows with more partitions and `--use_cache 0`; Baseline's full-attention ablation controls sequence length via `seq_len`.

### Key configuration flags

| Flag | Meaning |
|------|---------|
| `--use_cache 1` | Enable fixed-window duplicate-node/cache code path (standard mode; see KV cache caveat above) |
| `--use_cache 0` | Disable KV cache; enables the dynamic `node_out()` / `node_in()` path if loss stagnation is detected |
| `--attn_type sparse/full` | Sparse (edge-based) or full (dense) attention |
| `--use_preprocess_cache 0/1` | Cache/reuse the PPR+Metis preprocess result (only with `--use_cache 1`; default is 1, but most non-papers100M scripts pass 0 to force rebuild) |
| `--ppr_backend appnp/torch_geometric` | cuSPARSE GPU PPR vs PyG CPU PPR |
| `--n_parts` | Requested number of graph windows; should be even because construction makes 2 parents and `n_parts // 2` children per parent |
| `--related_nodes_topk_rate` | % of external neighbors merged into each window |
| `--struct_enc True` | Enable centrality + spatial positional encoding |
| `--preprocess_only 1` | Stop after preprocessing, before training |

## CUDA custom op

`core/ppr_backends/csrc/cusparse_spgemm.cpp` and `core/ppr_backends/csrc/cusparse_spgemm.cu` implement a cuSPARSE SpGEMM extension compiled via `torch.utils.cpp_extension.load()` in `cusparse_ops.py`. First run triggers JIT compilation (slow); subsequent runs reuse the cached `.so`.

## Current Implementation Notes

- `README.md` is stale and still references an Ascend/TorchGT setup; use this file and the scripts under `NeutronGT/scripts/` as the current guide.
- APPNP distributed PPR shards source nodes by rank, writes shard files under `dataset/<name>/ppr_temp/`, and rank 0 merges them from disk to reduce rank-0 memory pressure.
- Window bundle broadcast is also disk-backed via `dataset/<name>/window_state_cache/`, then rank 0 deletes the temporary bundle files.
- With `struct_enc=False` (the default), `StructInfo.sorted_ppr_matrix` is released after Metis construction to save memory; spatial position is only rebuilt when `--struct_enc True`.
