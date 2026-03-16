PPR + Metis 方案下的torchgt

bash run.sh ppr 0 8012 ogbn-papers100M graphormer True 5 ./vis 测试 Large Graphormer的 paper100M 性能，这个模型下对机器压力最大
想测别的模型记得在run.sh里改，use_cache 在 shell 里写的为 1，固定使用
