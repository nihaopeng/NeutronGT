def draw_heat_map(score, ids, prefix="full", topk=None, max_display=100, normalize=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    # 1. 确保转为 CPU numpy
    score_np = score.abs().cpu().detach().numpy()
    ids_np = ids.cpu().detach().numpy()
    print(f"📊 初始数据: {prefix.upper()} (score: {score_np.shape}, ids: {ids_np.shape})")
    # 2. 处理多头注意力 (Heads)
    if score_np.ndim == 3:
        # 建议取平均值更能代表整体注意力分布
        score_np = score_np[0]
    # 3. 核心：切片对齐
    num_nodes = len(ids_np)
    if score_np.shape[0] != num_nodes or score_np.shape[1] != num_nodes:
        score_np = score_np[:num_nodes, :num_nodes]
    # --- 新增：归一化处理 ---
    if normalize:
        # 防止除以 0，添加极小值 epsilon
        eps = 1e-8
        # 方式 A：按行归一化 (Row-wise Normalization)
        # 这样每一行的注意力权重总和为 1，突出每一行内最重要的节点
        row_sums = score_np.sum(axis=1, keepdims=True)
        score_np = score_np / (row_sums + eps)
        print("⚖️ 已完成按行归一化 (Row-wise Normalization)")  
        # 如果你想做全局 Min-Max 归一化，可以使用下面的代码：
        # score_np = (score_np - score_np.min()) / (score_np.max() - score_np.min() + eps)
    # -----------------------
    # --- Top-K 过滤逻辑 ---
    if topk is not None and topk < score_np.shape[1]:
        for i in range(score_np.shape[0]):
            row = score_np[i]
            mask_indices = np.argsort(row)[:-topk] 
            score_np[i, mask_indices] = 0
        print(f"✨ 已应用 Top-{topk} 过滤")
    # 4. 抽样：避免热力图像素过多
    if len(ids_np) > max_display:
        step = len(ids_np) // max_display
        score_np = score_np[::step, ::step]
        ids_np = ids_np[::step]
    # 5. 构建 DataFrame 并绘图
    df = pd.DataFrame(score_np, index=ids_np, columns=ids_np)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(15, 12))
    # 使用 robust=True 可以自动处理异常值点，让颜色映射更集中在主体数据
    # 如果已经归一化到 0-1，可以固定 vmin=0, vmax=1
    sns.heatmap(df,cbar=False, cmap='YlGnBu', robust=True, vmin=0 if normalize else None)
    # plt.title(f'Attention Score Heatmap ({prefix.upper()}) {"- Normalized" if normalize else ""}')
    # plt.xlabel('Target Node ID')
    # plt.ylabel('Source Node ID')
    plt.tight_layout()
    plt.savefig(f'attn_heatmap_{prefix}.png')
    plt.close()
    print(f"✅ 已保存热力图: attn_heatmap_{prefix}.png")
    
def draw_heat_map_binary(score, ids, prefix="full", topk=None, max_display=100, normalize=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd 
    # 1. 确保转为 CPU numpy
    score_np = score.abs().cpu().detach().numpy()
    ids_np = ids.cpu().detach().numpy() 
    print(f"📊 初始数据: {prefix.upper()} (score: {score_np.shape}, ids: {ids_np.shape})")  
    # 2. 处理多头注意力 (Heads)
    if score_np.ndim == 3:
        # 建议取平均值更能代表整体注意力分布
        score_np = score_np.mean(axis=0) 

    # 3. 核心：切片对齐
    num_nodes = len(ids_np)
    if score_np.shape[0] != num_nodes or score_np.shape[1] != num_nodes:
        score_np = score_np[:num_nodes, :num_nodes] 
    # --- 新增：归一化处理 ---
    if normalize:
        # 防止除以 0，添加极小值 epsilon
        eps = 1e-8
        # 方式 A：按行归一化 (Row-wise Normalization)
        # 这样每一行的注意力权重总和为 1，突出每一行内最重要的节点
        row_sums = score_np.sum(axis=1, keepdims=True)
        score_np = score_np / (row_sums + eps)
        print("⚖️ 已完成按行归一化 (Row-wise Normalization)")
                # 如果你想做全局 Min-Max 归一化，可以使用下面的代码：
        # score_np = (score_np - score_np.min()) / (score_np.max() - score_np.min() + eps)
    # -----------------------   
    # --- Top-K 过滤逻辑 ---
    if topk is not None and topk < score_np.shape[1]:
        for i in range(score_np.shape[0]):
            row = score_np[i]
            mask_indices = np.argsort(row)[:-topk] 
            score_np[i, mask_indices] = 0
        print(f"✨ 已应用 Top-{topk} 过滤") 
    # 4. 抽样：避免热力图像素过多
    if len(ids_np) > max_display:
        step = len(ids_np) // max_display
        score_np = score_np[::step, ::step]
        ids_np = ids_np[::step]

    # 5. 构建 DataFrame 并绘图
    df = pd.DataFrame(score_np, index=ids_np, columns=ids_np)   
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(15, 12))

    # 使用 robust=True 可以自动处理异常值点，让颜色映射更集中在主体数据
    # 如果已经归一化到 0-1，可以固定 vmin=0, vmax=1
    sns.heatmap(df,cbar=False, cmap='YlGnBu', robust=True, vmin=0 if normalize else None)

    # plt.title(f'Attention Score Heatmap ({prefix.upper()}) {"- Normalized" if normalize else ""}')
    # plt.xlabel('Target Node ID')
    # plt.ylabel('Source Node ID')
    plt.tight_layout()

    plt.savefig(f'attn_heatmap_{prefix}.png')
    plt.close()
    print(f"✅ 已保存热力图: attn_heatmap_{prefix}.png")