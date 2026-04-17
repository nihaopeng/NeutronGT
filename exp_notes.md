# TorchGT OverAll performance


## 配置参数
模型的配置参考了TorchGT的配置

| Model           | n_layers | hidden_dim | ffn_dim | num_heads |
| --------------- | -------- | ---------- | ------- | --------- |
| **GPH_{Slim}**  | 4        | 64         | 64      | 8         |
| **GT**          | 4        | 128        | 128     | 8         |
| **GPH_{Large}** | 12       | 768        | 768     | 32        |



具体超参数，ogbn-products、ogbn-arxiv、Amazon 都严格参照baseline的设计、reddit是自己定的，其seqlen较小时因为相较于其他dataset，reddit 有更高的 feature dimension，需要更多的显存



| Dataset       | GPH_{Slim}                         | GPH_{Large}                       | GT                                 |
| ------------- | ---------------------------------- | --------------------------------- | ---------------------------------- |
| ogbn-products | seq_len = 256K          EPOCHS=500 | seq_len = 32K          EPOCHS=200 | seq_len = 256K          EPOCHS=500 |
| ogbn-arxiv    | seq_len = 256K          EPOCHS=500 | seq_len = 32K         EPOCHS=200  | seq_len = 256K          EPOCHS=500 |
| Amazon        | seq_len = 256K          EPOCHS=500 | seq_len = 32K         EPOCHS=200  | seq_len = 256K          EPOCHS=500 |
| reddit        | seq_len = 32K          EPOCHS=500  | seq_len = 8K         EPOCHS=200   | seq_len = 32K          EPOCHS=500  |

## 处理 TorchGT 所需的数据集

1. 下载数据集

2. 处理成类似这样的结构
    
    ```
    ./dataset
    |__ogbn-arxiv
        |__ x.pt
        |__ y.pt
        |__ edge_index.pt
    ```
    
    


## 使用处理后的数据文件在 TorchGT 中进行测试

主要使用 main_sp_node_level.py 进行测试，具体的参数配置和 TorchGT 一致，在 utils/parser_node_level.py 内

我们可以使用仓库内的脚本，按如下方式测试

```shell
cd Baseline
conda activate gt
bash ./scripts/run_torchGT.sh 0,1,2,3 --arxiv --GT
```

其中 run_torchGT.sh 的选项格式为 bash $0 <devices> [dataset_flag] [model_flag]

可以在./TorchGT_logs/看到输出结果

```text
------------------------------------------------------------------------------------
Epoch: 001, Loss: 3.6889, Epoch Time: 0.635s, Full Epoch Time: 6.579s, Batch Prep Time: 5.931s
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Epoch: 002, Loss: 3.6889, Epoch Time: 0.360s, Full Epoch Time: 5.961s, Batch Prep Time: 5.593s
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Epoch: 003, Loss: 3.6889, Epoch Time: 0.268s, Full Epoch Time: 5.665s, Batch Prep Time: 5.391s
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Epoch: 004, Loss: 3.6888, Epoch Time: 0.219s, Full Epoch Time: 5.526s, Batch Prep Time: 5.301s
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Epoch: 005, Loss: 3.6887, Epoch Time: 0.189s, Full Epoch Time: 5.443s, Batch Prep Time: 5.248s
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Eval time 0.5465583801269531s
Epoch: 005, Loss: 3.688713, Train acc: 15.77%, Val acc: 15.71%, Test acc: 16.04%, Epoch Time: 0.189s, Full Epoch Time: 5.443s, Reorder Time: 5.248s
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Epoch: 006, Loss: 3.6886, Epoch Time: 0.173s, Full Epoch Time: 5.396s, Batch Prep Time: 5.218s
------------------------------------------------------------------------------------
```

可以观察baseline的实验效果



# NeutronGT OverAll performance

## 数据处理

我们需要手动将 COO 格式的 edge_index.pt 处理为 CSR 格式的边数据文件，命名为 edge_index_csr.pt，此时数据集应该是类似这样的结构

```
./dataset
|__ogbn-arxiv
    |__ x.pt
    |__ y.pt
    |__ edge_index.pt
    |__ edge_index_csr.pt
```



## 使用处理后的数据文件在 TorchGT 中进行测试

主要使用main_sp_node_level_ppr.py 及 core 中的文件

一定要使用12.1的cuda版本，否则预处理中的算子可能编译出错，可以使用以下命令

```shell
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
```

首次使用时会编译算子，耗时可能较长，具体测试的时间应该取后几次





采用的分区超参数设置

| dataset/model                | n_parts | relative_topk |
| ---------------------------- | ------- | ------------- |
| **AmazonProducts/GT**        | 128     | 4             |
| **AmazonProducts/GPH_Slim**  | 128     | 4             |
| **AmazonProducts/GTH_Large** | 400     | 4             |
| **ogbn-arxiv/GT**            | 16      | 8             |
| **ogbn-arxiv/GPH_Slim**      | 16      | 8             |
| **ogbn-arxiv/GTH_Large**     | 32      | 8             |
| **ogbn-products/GT**         | 128     | 6             |
| **ogbn-products/GPH_Slim**   | 128     | 6             |
| **ogbn-products/GTH_Large**  | 512     | 4             |
| **reddit/GT**                | 32      | 10            |
| **reddit/GPH_Slim**          | 32      | 10            |
| **reddit/GTH_Large**         | 80      | 4             |



```shell
cd NeutronGT
bash ./scripts/run_NeutronGT.sh 0,1,2,3  --arxiv --GT 
```

可以得到 NeutronGT 的实验效果




# ablation experiment

## Baseline



```shell
cd Baseline
bash ./scripts/run_ablation_1.sh 0,1,2,3  --arxiv --GT 
```

在 baseline 文件夹中，测试

full attention ，16000

即可得到消融实验中Baseline所代表的效果



## Basline+HAW

切换至 NeutronGT 文件夹，测试 full attention window，即可得到



```shell
cd NeutronGT
bash ./scripts/run_ablation_2.sh 0,1,2,3  --arxiv --GT 
```







## Baseline+HAW+RWP

在NeutronGT文件夹下，继续测试启用稀疏注意力与cache的效果

```shell
bash ./scripts/run_ablation_3.sh 0,1,2,3  --arxiv --GT 
```











# NeutronGT 500 epoch end2end time breakdown



按照之前的NeutronGT的overall方式，切换数据集和模型即可，例如

```shell
bash ./scripts/run_NeutronGT.sh 0,1,2,3  --arxiv --GPH_Large
```

即可看到 arxiv 数据集的预处理的两阶段的开销