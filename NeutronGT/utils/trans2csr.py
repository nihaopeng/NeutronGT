import torch
import os

# 1. 设置路径
import sys
dataset_name = sys.argv[1]
folder_path = f'./dataset/{dataset_name}'
edge_index_path = os.path.join(folder_path, 'edge_index.pt')
x_path = os.path.join(folder_path, 'x.pt')

# 2. 加载原始数据
edge_index = torch.load(edge_index_path)

# 获取节点总数 (N)
if os.path.exists(x_path):
    num_nodes = torch.load(x_path).size(0)
else:
    num_nodes = int(edge_index.max()) + 1

# 3. 转换为 CSR 格式
# 注意：edge_index [2, E] 需要先排序（按行排序）才能转 CSR
# 我们先转为普通的稀疏张量，再转为 CSR
row, col = edge_index
value = torch.ones(row.size(0)) # 假设权重为1

adj_coo = torch.sparse_coo_tensor(edge_index, value, size=(num_nodes, num_nodes))
adj_csr = adj_coo.to_sparse_csr()

data = {
    "rowptr": adj_csr.crow_indices(),
    "col": adj_csr.col_indices(),
    "value": adj_csr.values()
}

# 4. 保存为 .pt 文件
save_path = os.path.join(folder_path, 'edge_index_csr.pt')
torch.save(data, save_path)

print(f"转换成功！文件已保存至: {save_path}")
print(f"张量布局: {adj_csr.layout}") # 应该是 torch.sparse_csr