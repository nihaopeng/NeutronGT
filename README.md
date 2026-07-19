## TorchGT with PPR + METIS Partitioning

Run the following command to evaluate the **Graphormer-Large** model on the **ogbn-papers100M** dataset under the **PPR + METIS** partitioning scheme. This configuration places the highest memory and computational demand on the system.

```bash
bash run.sh ppr 0 8012 ogbn-papers100M graphormer True 5 ./vis
```

To evaluate other models, modify the model configuration in `run.sh`. The `use_cache` option is fixed to `1` in the script and should not be changed.

---

## Implementation Details

The primary modification is replacing the window-parallel indexing key from the **local partition index (`i`)** to the **global partition ID (`global_pid`)**.

Each rank first constructs `local_partition_ids` and `local_partitions`. During both training and evaluation, all partition-specific data—including `wm.sub_edge_index_for_partition_results`, `wm.dup_indices`, and `spatial_pos_by_pid`—is accessed using `global_pid`. This eliminates the partition mismatch that previously occurred in multi-GPU training, where nodes could become misaligned with their corresponding edges and structural encodings.

The evaluation pipeline has also been redesigned. Each rank first computes predictions for its local partitions, after which `dist.all_gather_object` is used to collect results from all ranks. Rank 0 then computes and reports the **global accuracy**, rather than the accuracy of its own local partitions.

During the window refinement stage, rank 0 gathers `scores_by_pid` from all ranks, performs the global `node_out` and `node_in` operations, and then broadcasts the updated window state back to every rank to ensure consistency across all processes.

In addition, a bug related to the scope of `parent_id` has been fixed. During the window expansion stage, the implementation now explicitly iterates over `self.child_partitions` using `enumerate(self.child_partitions)`, ensuring that `_merge_feature_sim(..., current_parent_id=parent_id)` always receives the correct parent partition ID corresponding to the current partition.
