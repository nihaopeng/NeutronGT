# TorchGT with PPR + METIS Partitioning

## Running

The following command evaluates the **Graphormer-Large** model on the **ogbn-papers100M** dataset using the **PPR + METIS** partitioning pipeline. This configuration represents the most demanding workload in terms of GPU memory and overall system resources.

```bash
bash run.sh ppr 0 8012 ogbn-papers100M graphormer True 5 ./vis
```

To evaluate a different model, modify the corresponding model configuration in `run.sh`. The `use_cache` option is fixed to `1` in the shell script and should remain enabled.

---

# Multi-GPU Window Parallelism

The major change is replacing the window-parallel indexing key from the **local partition index (`i`)** to the **global partition ID (`global_pid`)**.

Each GPU rank first constructs its own `local_partition_ids` and `local_partitions`. During both training and evaluation, partition-specific data—including

* `wm.sub_edge_index_for_partition_results`
* `wm.dup_indices`
* `spatial_pos_by_pid`

is accessed using `global_pid` instead of the local index. This eliminates the inconsistency previously observed in multi-GPU execution, where nodes could become misaligned with their corresponding edges and structural encodings.

The evaluation pipeline has also been redesigned. Each rank first collects prediction results for its local windows, and the results are then aggregated across all processes using `dist.all_gather_object`. Rank 0 computes and reports the **global accuracy** over the entire dataset instead of only evaluating its own local windows.

The window refinement stage is similarly synchronized. Rank 0 gathers `scores_by_pid` from all GPUs, performs the global `node_out` and `node_in` operations, and broadcasts the updated window state back to every rank, ensuring all processes maintain an identical partition layout.

A bug related to the scope of `parent_id` has also been fixed. During window expansion, the implementation now explicitly iterates over `self.child_partitions` using `enumerate(self.child_partitions)`, guaranteeing that

```python
_merge_feature_sim(..., current_parent_id=parent_id)
```

always receives the correct parent partition ID.

---

# FORA Integration

The following upstream FORA source files have been modified:

```text
third_party/fora/fora.cpp
third_party/fora/query.h
third_party/fora/config.h
third_party/fora/CMakeLists.txt
```

## Changes

### Dumping Top-k Results

The original FORA implementation does not export top-k PPR results in a format directly consumable by NeutronGT.

A new command-line option

```text
--dump_topk_path
```

has been added, allowing FORA to dump

```
source destination score
```

triples directly to disk.

### Configurable Teleport Probability

A new option

```text
--alpha
```

has been introduced so that the project's `--ppr_alpha` argument is correctly forwarded to FORA.

### Build Compatibility

The build system has been updated to support locally vendored Boost releases.

The modifications include:

* compatibility with Boost 1.87
* removal of hard-coded `-lboost_*` libraries
* standard CMake Boost targets
* compatibility with deprecated `boost/progress.hpp` via `BOOST_TIMER_ENABLE_DEPRECATED`

---

# PPR Backend

The original `personal_pagerank()` implementation in

```text
core/pprPartition.py
```

has been refactored into two interchangeable backends:

* `torchgeo`
* `fora`

The FORA backend additionally implements

* graph export
* FORA binary invocation
* top-k result parsing
* PPR result caching

The returned interface remains unchanged:

```python
(edge_index, ppr_val)
```

Therefore, the downstream pipeline remains identical:

```text
add_isolated_connections()
        ↓
build_adj_fromat()
        ↓
weightMetis_keepParent()
```

No modifications are required in the graph partitioning stage.

---

# New Runtime Arguments

The following command-line options have been added.

| Argument                  | Description                                                                                                                  |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `--ppr_backend`           | Selects the PPR backend (`torchgeo` or `fora`).                                                                              |
| `--ppr_topk`              | Number of retained PPR neighbors for each source node. Larger values generate denser PPR graphs and larger METIS partitions. |
| `--ppr_alpha`             | Teleport probability used in Personalized PageRank. Default: `0.85`.                                                         |
| `--ppr_cache_dir`         | Directory for cached PPR results. Defaults to `dataset/<dataset>/`.                                                          |
| `--fora_bin`              | Path to the FORA executable. Default: `./third_party/fora/build/fora`.                                                       |
| `--fora_work_dir`         | Working directory containing exported graph files and intermediate outputs.                                                  |
| `--fora_epsilon`          | Approximation parameter of FORA. Smaller values improve accuracy at the cost of runtime.                                     |
| `--fora_query_batch_size` | Number of query nodes processed by FORA in one batch. `0` processes the entire graph in a single run.                        |

---

# Graph Export Utility

A new utility

```text
scripts/export_graph_for_fora.py
```

exports `edge_index.pt` into the FORA input format:

* `graph.txt`
* `attribute.txt`
* `ssquery.txt`

---

# Overall Data Flow

The complete execution pipeline is

```
run.sh
    ↓
torchrun main_sp_node_level_ppr.py
    ↓
build_graph_struct_info()
    ↓
personal_pagerank(..., backend="fora")
    ↓
load cache or invoke FORA
    ↓
(edge_index, ppr_val)
    ↓
add_isolated_connections()
    ↓
build_adj_fromat()
    ↓
weightMetis_keepParent()
    ↓
window construction
    ↓
spatial encoding
    ↓
distributed window-parallel training
```

When the FORA backend is selected:

1. Check whether

```
dataset/<dataset>/ppr_fora_topk{K}_alpha{A}.pt
```

already exists.

2. If present, load the cached `(edge_index, ppr_val)`.

3. Otherwise,

* export the graph,
* invoke the FORA executable,
* parse the dumped top-k results,
* construct `(edge_index, ppr_val)`,
* cache the tensor for future runs.

The remaining METIS partitioning and window construction pipeline is unchanged.

---

# Multi-process FORA Data Flow

The parallel FORA implementation follows the pipeline below.

```
Select query nodes
        ↓
Export shared graph
        ↓
Split query nodes into batches
        ↓
Create one FORA workspace per batch
        ↓
Launch FORA workers in parallel
        ↓
Collect batch TSV files
        ↓
Merge PPR tensors
        ↓
Write final cache
```

## 1. Query Node Selection

Query nodes are determined in `personal_pagerank_fora()`.

If

```
ppr_high_degree_ratio < 1
```

only the highest-degree nodes are selected.

Otherwise, all graph nodes are used.

---

## 2. Shared Graph Export

The graph is exported only once into

```
third_party/fora/data/<dataset>/
```

including

```
graph.txt
attribute.txt
```

If CSR data are available, they are exported directly from CSR; otherwise, the original edge list is used.

This directory is read-only and shared by all workers.

---

## 3. Batch Construction

The query nodes are split according to

```
fora_query_batch_size
```

forming batches such as

```
batch_0
batch_10000
batch_20000
...
```

Each batch becomes one independent task.

---

## 4. Independent FORA Workspaces

Each batch creates its own dataset directory, for example,

```
third_party/fora/data/ogbn-arxiv__batch_0
third_party/fora/data/ogbn-arxiv__batch_10000
```

The shared graph files are linked into the batch directory, while each batch independently writes

```
ssquery.txt
topk_batch_<start>.tsv
```

This design guarantees that concurrent workers never overwrite one another.

---

## 5. Parallel FORA Execution

If

```
fora_num_workers == 1
```

batches are executed sequentially.

Otherwise,

```
multiprocessing.get_context("spawn").Pool(...)
```

is used to execute `_run_fora_topk_worker()` in parallel.

Each worker simply launches one FORA process for its assigned batch.

---

## 6. FORA Worker

Inside each worker,

1. `ssquery.txt` is generated.
2. The external FORA executable is launched.
3. FORA reads

```
graph.txt
attribute.txt
ssquery.txt
```

4. FORA produces

```
topk_batch_<start>.tsv
```

Standard output from FORA is suppressed to keep training logs clean.

---

## 7. Result Collection

After all workers finish,

the main process traverses every batch in order,

parses each

```
topk_batch_*.tsv
```

into

```
edge_index
edge_value
```

and appends them to

```
edge_parts
value_parts
```

---

## 8. Cache Generation

Finally,

```python
torch.cat(edge_parts)
torch.cat(value_parts)
```

construct the complete PPR graph.

The final cache

```
ppr_fora_*.pt
```

stores both the PPR tensors and metadata, including

* query mode
* query ratio
* number of query nodes
* query policy

allowing subsequent runs to bypass the expensive PPR computation entirely.
