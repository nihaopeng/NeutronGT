import torch


def save_model_output(args, score, global_ids, N,acc,prefix="full"):
    path_score = f"{prefix}_batch_{args.dataset}.pt"
    
    score_cpu = score.detach().cpu()
    ids_cpu = global_ids.detach().cpu()
    
    # 1. 移除 Padding，获取有效节点及其对应的 ID
    valid_mask = ids_cpu >= 0
    final_ids = ids_cpu[valid_mask]
    
    # 2. 截取有效的注意力矩阵 [Heads, Valid_N, Valid_N]
    final_score = score_cpu[:, valid_mask, :][:, :, valid_mask]
    
    # 3. 两个文件成对保存
    acc_with_high_score = {
        "final_score":final_score,
        "final_ids":final_ids,
        "acc":acc
    }
    torch.save(acc_with_high_score, path_score)
        
    print(f"✅ [{prefix.upper()}] 已保存 Score ({final_score.shape}) 和 ID ({final_ids.shape})")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

def analyze_full_vs_mini(dataset_name, top_ratio=0.1):
    # 路径定义
    full_path = f"full_batch_{dataset_name}.pt"
    mini_path = f"mini_batch_{dataset_name}.pt"

    # 1. 加载数据
    full_data = torch.load(full_path, map_location='cpu')
    mini_data = torch.load(mini_path, map_location='cpu')

    print(f"📊 数据读取成功: {dataset_name}")
    print(f"Full-batch Acc: {full_data['acc']:.4f}")
    print(f"Mini-batch Acc: {mini_data['acc']:.4f}")

    # 2. 提取 Full-batch 的 Top-K 全局边
    full_score = full_data['final_score'].mean(0) # [N, N]
    full_ids = full_data['final_ids']
    
    k_full = int(full_score.numel() * top_ratio)
    _, topk_indices = torch.topk(full_score.view(-1), k_full)
    
    full_n = full_score.size(1)
    full_top_edges = set()
    for idx in topk_indices.tolist():
        r, c = idx // full_n, idx % full_n
        full_top_edges.add((full_ids[r].item(), full_ids[c].item()))

    # 3. 提取 Mini-batch 的 Top-K 全局边
    mini_score = mini_data['final_score'].mean(0) # [M, M]
    mini_ids = mini_data['final_ids']
    
    k_mini = int(mini_score.numel() * top_ratio)
    _, mini_indices = torch.topk(mini_score.view(-1), k_mini)
    
    mini_n = mini_score.size(1)
    mini_top_edges = set()
    for idx in mini_indices.tolist():
        r, c = idx // mini_n, idx % mini_n
        mini_top_edges.add((mini_ids[r].item(), mini_ids[c].item()))

    # 4. 计算交集捕捉率
    hits = full_top_edges.intersection(mini_top_edges)
    capture_rate = len(hits) / len(full_top_edges) if full_top_edges else 0

    print("-" * 30)
    print(f"🏆 关键边捕捉率 (Recall): {capture_rate:.4f}")
    print(f"📉 准确率损耗 (Acc Drop): {full_data['acc'] - mini_data['acc']:.4f}")
    
    return capture_rate