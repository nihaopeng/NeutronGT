import os
import torch
import numpy as np
from torch_geometric.datasets import AmazonProducts
from torch_geometric.utils import coalesce

_original_np_load = np.load
def _patched_np_load(*args, **kwargs):
    kwargs['allow_pickle'] = True
    return _original_np_load(*args, **kwargs)
np.load = _patched_np_load

def download_and_prepare_amazon(root_dir='./dataset'):
    print("1. 初始化并修复 PyG 目录结构...")
    pyg_root = os.path.join(root_dir, 'AmazonProducts')
    os.makedirs(os.path.join(pyg_root, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(pyg_root, 'processed'), exist_ok=True)
    
    print("2. 触发 PyG 下载与内部预处理...")
    dataset = AmazonProducts(root=pyg_root)
    data = dataset[0]
    
    print("3. 提取特征与原始拓扑数据...")
    x = data.x
    y = data.y
    
    if y.dim() > 1 and y.shape[1] > 1:
        y = torch.argmax(y, dim=-1)
    
    edge_index = data.edge_index
    edge_index = coalesce(edge_index, num_nodes=x.size(0))
    
    print("4. 持久化保存为标准 .pt 文件...")
    output_dir = os.path.join(root_dir, 'AmazonProducts')
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(x, os.path.join(output_dir, 'x.pt'))
    torch.save(y, os.path.join(output_dir, 'y.pt')) 
    torch.save(edge_index, os.path.join(output_dir, 'edge_index.pt'))
    
    print("====== 纯净版下载与提取完毕 ======")
    print(f"节点总数: {x.size(0)}")
    print(f"特征维度: {x.size(1)}")
    print(f"有效边数: {edge_index.size(1)}")
    print(f"类别数: {y.max().item() + 1}")
    print(f"数据集已输出至: {output_dir}")
    print("==================================")

if __name__ == "__main__":
    download_and_prepare_amazon()
