# torchgt on ascend npu

## ascend 910B4

## run
torchgt

cd TorchGT

bash scripts/1_efficiency.sh

## torch_npu
bash run.sh


bash run.sh 0 cora，如果不存在cora_reordered_train_idx.pt文件就会从零跑
如果存在就会从文件中加载reorder后的idx
