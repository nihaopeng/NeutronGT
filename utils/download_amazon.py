import os
import torch
import numpy as np
from torch_geometric.datasets import AmazonProducts
from torch_geometric.utils import coalesce

# ====== 核心补丁：动态劫持 np.load 强制允许 pickle ======
_original_np_load = np.load
def _patched_np_load(*args, **kwargs):
    kwargs['allow_pickle'] = True
    return _original_np_load(*args, **kwargs)
np.load = _patched_np_load
# =========================================================

def prepare_amazon_pt(root_dir='./dataset'):
    print("开始下载并处理 AmazonProducts 数据集，这可能需要一些时间...")
    # 1. 自动下载并构建数据集
    dataset = AmazonProducts(root=os.path.join(root_dir, 'AmazonProducts'))
    data = dataset[0]
    
    print("数据加载完成，正在提取特征...")
    # 2. 提取核心张量
    x = data.x           # 特征矩阵: [num_nodes, 200]
    y = data.y           # 标签矩阵: [num_nodes]
    edge_index = data.edge_index # 边索引: [2, num_edges]
    
    # 确保边索引是标准化的（无重复边、内部有序）
    # 这对后续传入 torchgt 的稀疏注意力计算非常重要
    edge_index = coalesce(edge_index, num_nodes=x.size(0))
    
    # 3. 创建输出目录
    output_dir = os.path.join(root_dir, 'amazon_pt')
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. 独立保存为 .pt 文件
    x_path = os.path.join(output_dir, 'x.pt')
    t_path = os.path.join(output_dir, 't.pt') # 保存为目标 t.pt 或 y.pt
    edge_path = os.path.join(output_dir, 'edge_index.pt')
    
    torch.save(x, x_path)
    torch.save(y, t_path) 
    torch.save(edge_index, edge_path)
    
    print("====== 处理完毕 ======")
    print(f"节点数量 (Nodes): {x.size(0)}")
    print(f"特征维度 (Features): {x.size(1)}")
    print(f"类别数量 (Classes): {dataset.num_classes}")
    print(f"边数量 (Edges): {edge_index.size(1)}")
    print(f"文件已持久化保存至: {output_dir}")
    print("======================")

if __name__ == "__main__":
    prepare_amazon_pt()