# TorchGT Overall Performance

## Configuration

The model configurations follow the original TorchGT settings.

| Model         | n_layers | hidden_dim | ffn_dim | num_heads |
| ------------- | -------: | ---------: | ------: | --------: |
| **GPH_Slim**  |        4 |         64 |      64 |         8 |
| **GT**        |        4 |        128 |     128 |         8 |
| **GPH_Large** |       12 |        768 |     768 |        32 |

For detailed hyperparameters:

* **ogbn-products**, **ogbn-arxiv**, and **Amazon** strictly follow the baseline settings.
* **reddit** uses custom settings. Since Reddit has a higher feature dimension than the other datasets, a shorter sequence length is used to reduce GPU memory consumption.

| Dataset       | GPH_Slim                     | GPH_Large                   | GT                           |
| ------------- | ---------------------------- | --------------------------- | ---------------------------- |
| ogbn-products | seq_len = 256K, EPOCHS = 500 | seq_len = 32K, EPOCHS = 200 | seq_len = 256K, EPOCHS = 500 |
| ogbn-arxiv    | seq_len = 256K, EPOCHS = 500 | seq_len = 32K, EPOCHS = 200 | seq_len = 256K, EPOCHS = 500 |
| Amazon        | seq_len = 256K, EPOCHS = 500 | seq_len = 32K, EPOCHS = 200 | seq_len = 256K, EPOCHS = 500 |
| reddit        | seq_len = 32K, EPOCHS = 500  | seq_len = 8K, EPOCHS = 200  | seq_len = 32K, EPOCHS = 500  |

## Preparing Datasets for TorchGT

1. Download the dataset.

2. Organize the processed files into the following structure:

```text
./dataset
└── ogbn-arxiv
    ├── x.pt
    ├── y.pt
    └── edge_index.pt
```

## Running TorchGT

Use `main_sp_node_level.py` for evaluation. The parameter settings are identical to TorchGT and are defined in `utils/parser_node_level.py`.

Run:

```bash
cd Baseline
conda activate gt
bash ./scripts/run_torchGT.sh 0,1,2,3 --arxiv --GT
```

Script usage:

```bash
bash run_torchGT.sh <devices> [dataset_flag] [model_flag]
```

Logs are written to `./TorchGT_logs/`.

`Full Epoch Time` represents the end-to-end training time per epoch.

# NeutronGT Overall Performance

## Data Preparation

Convert the COO-format `edge_index.pt` into CSR format and save it as `edge_index_csr.pt`.

```text
./dataset
└── ogbn-arxiv
    ├── x.pt
    ├── y.pt
    ├── edge_index.pt
    └── edge_index_csr.pt
```

## Running NeutronGT

The main entry is `main_sp_node_level_ppr.py`.

CUDA **12.1** is required; otherwise, preprocessing operators may fail to compile.

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
```

The first execution compiles custom operators and may take significantly longer. Use subsequent runs for performance evaluation.

### Partition Hyperparameters

| Dataset / Model            | n_parts | related_nodes_topk |
| -------------------------- | ------: | -----------------: |
| AmazonProducts / GT        |     128 |                  4 |
| AmazonProducts / GPH_Slim  |     128 |                  4 |
| AmazonProducts / GPH_Large |     400 |                  4 |
| ogbn-arxiv / GT            |      16 |                  8 |
| ogbn-arxiv / GPH_Slim      |      16 |                  8 |
| ogbn-arxiv / GPH_Large     |      32 |                  8 |
| ogbn-products / GT         |     128 |                  6 |
| ogbn-products / GPH_Slim   |     128 |                  6 |
| ogbn-products / GPH_Large  |     512 |                  4 |
| reddit / GT                |      32 |                 10 |
| reddit / GPH_Slim          |      32 |                 10 |
| reddit / GPH_Large         |      80 |                  4 |

Run:

```bash
cd ../NeutronGT
bash ./scripts/run_NeutronGT.sh 0,1,2,3 --arxiv --GT
```

`Train Time` represents the end-to-end training time per epoch.

# Ablation Study

The ablation study is conducted on the **GPH_Slim** model under three settings:

1. Sequence parallelism with **full attention** (`seq_len = 16000`).
2. NeutronGT with **window-based parallelism** and full attention.
3. NeutronGT with **window-based parallelism**, **efficient sparse attention**, and **feature/KV-cache optimization**.

### Hyperparameter Settings

| Dataset        | Baseline                      | Baseline + HAW                           | Baseline + HAW + RWP                       |
| -------------- | ----------------------------- | ---------------------------------------- | ------------------------------------------ |
| reddit         | Full attention, seq_len = 16K | Full attention, 64 partitions, topk = 10 | Sparse attention, 32 partitions, topk = 10 |
| AmazonProducts | Full attention, seq_len = 16K | Full attention, 320 partitions, topk = 4 | Sparse attention, 128 partitions, topk = 4 |
| ogbn-arxiv     | Full attention, seq_len = 16K | Full attention, 56 partitions, topk = 8  | Sparse attention, 16 partitions, topk = 8  |
| ogbn-products  | Full attention, seq_len = 16K | Full attention, 600 partitions, topk = 6 | Sparse attention, 128 partitions, topk = 6 |

## Baseline

```bash
cd ../Baseline
bash ./scripts/run_ablation_1.sh 0,1,2,3 --arxiv --GPH_Slim
```

Evaluates the baseline configuration using full attention with `seq_len = 16000`.

## Baseline + HAW

```bash
cd ../NeutronGT
bash ./scripts/run_ablation_2.sh 0,1,2,3 --arxiv --GPH_Slim
```

Evaluates window-based full attention.

## Baseline + HAW + RWP

```bash
bash ./scripts/run_ablation_3.sh 0,1,2,3 --arxiv --GPH_Slim
```

Evaluates sparse attention with cache enabled.

# NeutronGT 500-Epoch End-to-End Time Breakdown

The hyperparameter configuration is the same as the **Overall Performance** evaluation for **GPH_Large**.

Example:

```bash
bash ./scripts/run_NeutronGT.sh 0,1,2,3 --arxiv --GPH_Large
```

This reports the two-stage preprocessing overhead and the training time for the selected dataset.
