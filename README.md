PPR + Metis 方案下的torchgt

bash run.sh ppr 0 8012 ogbn-papers100M graphormer True 5 ./vis 测试 Large Graphormer的 paper100M 性能，这个模型下对机器压力最大
想测别的模型记得在run.sh里改，use_cache 在 shell 里写的为 1，固定使用


---

核心修改是把窗口并行的主键从“本地下标 i”改成“全局窗口 ID global_pid”。现在每个 rank 先构造 local_partition_ids 和 local_partitions，训练与评估里都按 global_pid 取 wm.sub_edge_index_for_partition_results、wm.dup_indices、spatial_pos_by_pid，避免了之前多卡时节点和边/结构编码错位的问题。

评估也改成了每个 rank 收集本地窗口结果，再用 dist.all_gather_object 做全局汇总，rank 0 输出的是全局准确率而不是只看自己窗口。窗口调整阶段改成 rank 0 汇总各卡的 scores_by_pid 后统一执行 node_out/node_in，再把更新后的窗口状态广播回所有 rank。

另外修了 parent_id 作用域错误：现在扩窗阶段会显式 enumerate(self.child_partitions)，保证 _merge_feature_sim(..., current_parent_id=parent_id) 用的是当前父分区的真实 ID。



---

改过的 FORA 源码

third_party/fora/fora.cpp
third_party/fora/query.h
third_party/fora/config.h
third_party/fora/CMakeLists.txt

改动
- 加 --dump_topk_path
    upstream FORA 默认不会把 top-k 结果写成你项目能直接读的文件
    我补了一个输出路径，让 FORA 把 src dst score 落盘
- 加 --alpha
    让你的 --ppr_alpha 能真正传到 FORA
- 改构建兼容性
    让它能用你本地 vendoring 的 Boost 发布包编译
    处理 Boost 1.87 的废弃头兼容
    去掉旧的硬编码 -lboost_*

---

在 core/pprPartition.py 里把原来的 personal_pagerank() 拆成了 torchgeo 和 fora 两个 backend，并新增了 FORA 路径需要的几件事：
- 图导出
- FORA 二进制调用
- top-k 结果回读
- PPR 结果缓存

返回格式仍然是原来的 (edge_index, ppr_val)，所以后面的 add_isolated_connections -> build_adj_fromat -> weightMetis_keepParent 没动

在 main_sp_node_level_ppr.py 里把训练入口改成通过参数选择 PPR backend，在 utils/parser_node_level.py 里新增了 

--ppr_backend:选 PPR 计算后端,torchgeo 用原来的 PyG get_ppr;fora 用你现在接进去的 FORA
--ppr_topk: 每个源点最终只保留多少个 PPR 邻居. 这个值越大，PPR 图越稠密，后面 METIS 和窗口也会更大
--ppr_alpha PPR 的 teleport probability ,你之前主路径默认就是 0.85
--ppr_cache_dir PPR 结果缓存目录，不写时默认缓存到 dataset/<name>/，会存成类似 ppr_fora_topk5_alpha0p85.pt
--fora_bin FORA 可执行文件路径，现在默认就是：./third_party/fora/build/fora
--fora_work_dir FORA 工作目录，用来放导出的 graph.txt、attribute.txt、ssquery.txt 和 batch dump；默认：./third_party/fora/data
--fora_epsilon FORA 的近似误差参数，越小通常越准，但越慢
--fora_query_batch_size 一次喂给 FORA 多少个源点做 top-k，0 表示整图节点一次跑完，大图上如果内存紧，可以改成分批


新增了 scripts/export_graph_for_fora.py，把 edge_index.pt 导成 FORA 需要的 graph.txt、attribute.txt、ssquery.txt。

现在的数据流

1. 执行 run.sh。
2. ppr 分支用 torchrun 启动 main_sp_node_level_ppr.py。
3. 在 build_graph_struct_info() 里，会调用 core/pprPartition.py 的：personal_pagerank(..., backend="fora", ...)
    1. 先看缓存是否存在：dataset/<name>/ppr_fora_topk{K}_alpha{A}.pt
    2. 如果缓存存在，直接读缓存，得到 (edge_index, ppr_val)
    3. 如果缓存不存在，把原图导出到：
        third_party/fora/data/<dataset>/graph.txt
        third_party/fora/data/<dataset>/attribute.txt
        third_party/fora/data/<dataset>/ssquery.txt
        并调用： third_party/fora/build/fora topk ...
        FORA 把每个源点的 top-k PPR 结果写到 batch dump 文件
        Python 再把这些 dump 解析回 (edge_index, ppr_val)
        最后存成项目自己的缓存 .pt
4. 拿到 (edge_index, ppr_val) 之后，后面的图划分流程和你原先设计保持一致：
    add_isolated_connections()
    build_adj_fromat()
    weightMetis_keepParent()
5. weightMetis_keepParent() 产出窗口划分、重复节点、子图边等信息。
6. 如果开了结构编码，还会按窗口计算 spatial_pos。
7. 训练阶段每张卡只拿自己负责的窗口，继续走你现在的窗口并行训练逻辑。


> [!TIP] FORA 与 third_party 依赖
> 我把 FORA vendoring 到了 third_party/fora，并在 third_party/fora/UPSTREAM.md 记录了 upstream 来源和 commit。
> 因为 upstream FORA 默认不会把 top-k 结果写成项目可消费的文件，我在 third_party/fora/query.h、third_party/fora/fora.cpp、third_party/fora/config.h 里加了最小补丁：支持 --dump_topk_path 输出 top-k 结果，也支持 --alpha 透传 teleport probability。
> 
> 依赖这块，最开始我试过 boost superproject + submodule 方案，但那条路在 Jam 依赖闭包上过于脆弱，所以我改成了本地发布包方案：下载官方 Boost 1.87.0 发布包到 third_party/boost_release，在本地编出 filesystem/date_time/serialization，然后让 FORA 固定链接这份发布包。为此我改了 scripts/build_fora.sh 和 third_party/fora/CMakeLists.txt，去掉了旧的硬编码 -lboost_*，改成走 CMake 正常的 Boost target，并加了 BOOST_TIMER_ENABLE_DEPRECATED 来兼容 FORA 旧代码对 boost/progress.hpp 的依赖。

