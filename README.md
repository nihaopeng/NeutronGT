# torchgt on ascend npu

## ascend 910B4

## run
torchgt

cd TorchGT

bash scripts/1_efficiency.sh

## torch_npu
bash run.sh


bash run.sh 0 cora，if cora_reordered_train_idx.pt not exist， code will run from beginning

if exist， it will load reorder idx from disk.
