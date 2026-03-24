PPR + Metis 方案下的torchgt

bash run.sh ppr 0 8012 ogbn-papers100M graphormer True 5 ./vis 测试 Large Graphormer的 paper100M 性能，这个模型下对机器压力最大
想测别的模型记得在run.sh里改，use_cache 在 shell 里写的为 1，固定使用


---

核心修改是把窗口并行的主键从“本地下标 i”改成“全局窗口 ID global_pid”。现在每个 rank 先构造 local_partition_ids 和 local_partitions，训练与评估里都按 global_pid 取 wm.sub_edge_index_for_partition_results、wm.dup_indices、spatial_pos_by_pid，避免了之前多卡时节点和边/结构编码错位的问题。

评估也改成了每个 rank 收集本地窗口结果，再用 dist.all_gather_object 做全局汇总，rank 0 输出的是全局准确率而不是只看自己窗口。窗口调整阶段改成 rank 0 汇总各卡的 scores_by_pid 后统一执行 node_out/node_in，再把更新后的窗口状态广播回所有 rank。

另外修了 parent_id 作用域错误：现在扩窗阶段会显式 enumerate(self.child_partitions)，保证 _merge_feature_sim(..., current_parent_id=parent_id) 用的是当前父分区的真实 ID。

