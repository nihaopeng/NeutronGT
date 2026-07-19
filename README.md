# torchgt on ascend npu

## ascend 910B4

## dataset

`python utils/preprocess_data.py [dataset_name](e.g. python utils/preprocess_data.py cora)`

## torch_npu

`bash run.sh origin 0,1 cora True 5 vis_dir 8080`[code version][GPUs][dataset][is enable position embedding][max path distance][visual data path][port]

## install
## install visual font
`sudo apt install fonts-wqy-microhei`
*`rm -rf ~/.cache/matplotlib/`

## install torch
install torch, torch_npu (租的服务器已经安装has been installed on renting server，version:1.11)

## install dependencies
`pip install torch-geometric==2.0.0 torch-scatter==2.1.1 torch-summary==1.4.5 dgl==1.0.1 -i https://pypi.org/simple`

`pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html`

`pip install numba,pymetis`
