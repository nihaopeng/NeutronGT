# torchgt on ascend npu

## ascend 910B4

## dataset

`python utils/preprocess_data.py [dataset_name](e.g. python utils/preprocess_data.py cora)`

## torch_npu

`bash run.sh origin 0,1 cora True 5 vis_dir 8080`[代码版本][启动的GPU][数据集][是否启用位置编码][最大路径距离][可视化数据存放文件夹路径][port]

## install
## 安装可视化字体
`sudo apt install fonts-wqy-microhei`
*`rm -rf ~/.cache/matplotlib/`

## 安装torch
install torch, torch_npu (租的服务器已经安装，1.11版本)

## 安装依赖包
`pip install torch-geometric==2.0.0 torch-scatter==2.1.1 torch-summary==1.4.5 dgl==1.0.1 -i https://pypi.org/simple`

`pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html`

`pip install numba,pymetis`
## tip
1, 当前版本请不要使用除了full以外的attn type，他们都没有返回score，也不被我们需要