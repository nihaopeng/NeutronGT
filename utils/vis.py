from collections import Counter
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import networkx as nx
from tqdm import tqdm

from utils.logger import log

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

score_hist_flist = []
score_neighbor_ratio_list = []
score_neighbor_ratio_in_neighbor_list = []
score_relativity_ratio_list = []
high_attn_node_neighbor_neighbor_num = []
node_attned_list = []
pers = [95,90,80,70,60,40,20]
epochs = []
train_acc = []
val_acc = []
test_acc = []

vis_dir = ""

def pics_to_gif(flist,output_gif, frame_duration=100, loop=0):
    """
    读取文件夹中的图片生成GIF
    :param folder_path: 图片文件夹路径
    :param output_gif: 输出GIF的路径及文件名
    :param frame_duration: 每帧持续时间（毫秒），数值越小播放越快
    :param loop: 循环次数，0表示无限循环
    """
    # 获取文件夹中所有png/jpg格式的图片，并按文件名排序
    frames = []
    for img_path in flist:
        with Image.open(img_path) as img:
            # 转换为RGB模式，避免GIF不支持透明度导致的异常
            frames.append(img.convert('RGB'))
    # 保存为GIF
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=loop
    )
    log(f"GIF已保存至：{output_gif}")

def plot(x,y,label,dir,title,fig_name):
    fig, ax = plt.subplots(figsize=(12, 7))
    for l_idx, l in enumerate(label):
        y_i = y[l_idx]
        ax.plot(x, y_i,
                label=l)
    ax.legend(
        loc='upper right',
        fontsize=10,
        ncol=2,
        frameon=True,
        shadow=True
    )
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticks(epochs)
    plt.title(f"{title}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f'{dir}/{fig_name}', dpi=300, bbox_inches='tight')
    plt.close()

def hist(y,x_label,y_label,title,dir,file_name):
    plt.figure(figsize=(10, 6))
    n_bins = 50
    plt.hist(
        y,
        bins=n_bins,
        alpha=0.8,
        edgecolor="black",
        color="#1f77b4"
    )
    plt.title(f"{title})", fontsize=14, fontweight="bold")
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.tight_layout()
    fig_full_path = os.path.join(vis_dir,dir,f"{file_name}")
    if not os.path.exists(f"./{vis_dir}/{dir}"):
        os.makedirs(f"./{vis_dir}/{dir}")
    plt.savefig(fig_full_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_full_path

def bar(categories,values,title,dir,file_name):
    plt.bar(
        categories,
        values,
        alpha=0.8,
        edgecolor="black",
        color="#1f77b4",
        width=0.8
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("距离", fontsize=12)
    plt.ylabel("频数", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if not os.path.exists(f"./{vis_dir}/{dir}/"):
        os.makedirs(f"./{vis_dir}/{dir}/")
    fig_full_path = os.path.join(vis_dir,dir,file_name)
    plt.savefig(fig_full_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_full_path

def scatter(x,y,x_label,y_label,title,dir,file_name):
    if not isinstance(x[0], list):
        x,y = [x],[y]
    n_groups = len(x)
    default_colors = plt.cm.tab10(np.linspace(0, 1, min(n_groups, 10)))
    colors = default_colors[:n_groups]
    labels = [f'Group {i+1}' for i in range(n_groups)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_groups):
        scatter = ax.scatter(
            x[i], 
            y[i], 
            s=10,           # 点的大小
            c=[colors[i]],
            label=labels[i]  # 标签
        )
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    if not os.path.exists(f"./{vis_dir}/{dir}/"):
        os.makedirs(f"./{vis_dir}/{dir}/")
    fig_full_path = os.path.join(vis_dir,dir,file_name)
    plt.tight_layout()
    plt.savefig(fig_full_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_full_path


def high_attn(score_matri:np.ndarray,epoch:int):
    "所有的分数的直方图"
    plt.figure(figsize=(10, 6))
    n_bins = 50
    attention_flat = score_matri.flatten()
    hist(attention_flat,"注意力分数","频数","注意力分数分布直方图","score_hist",f"attention_histogram_{epoch}.png")

def neighbor_high_attn(score_matri:np.ndarray,edge_index:np.ndarray,idx:np.ndarray,epoch:int):
    "邻居的分数的直方图"
    # 步骤1：构建真实邻居边集合（无序，避免重复）
    edge_set = set()
    for u, v in edge_index.T:
        edge_set.add((min(u, v), max(u, v)))
    # 步骤2：筛选score_agg中属于真实邻居的分数
    neighbor_scores = []
    M = score_matri.shape[0]
    for i in range(M):
        for j in range(M):
            if i == j:  # 排除自环（节点对自身无邻居关系）
                continue
            # 获取score_agg[i][j]对应的真实节点ID
            u = idx[i]
            v = idx[j]
            # 判断是否为真实邻居
            edge_tuple = (min(u, v), max(u, v))
            if edge_tuple in edge_set:
                neighbor_scores.append(score_matri[i][j])
    # 转换为numpy数组（方便后续计算）
    neighbor_scores = np.array(neighbor_scores)
    plt.figure(figsize=(10, 6))
    n_bins = 50
    hist(neighbor_scores,"邻居对的注意力分数","频数","邻居注意力分数分布直方图","neighbor_score_hist",f"attention_histogram_{epoch}.png")

def neighbor(score_matri:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int):
    "高分中邻居占比、邻居中高分邻居的占比，邻居中"
    score_flat=score_matri.flatten()
    thresh_holds=np.percentile(score_flat,pers)
    edge_set = set()
    for u, v in edge_index.T:
        edge_set.add((min(u, v), max(u, v)))
    # 步骤2：构建节点-邻居映射（核心：快速查询节点的邻居数）
    node_neighbors = {}  # key: 节点ID, value: 该节点的邻居集合
    for u, v in edge_index.T:
        if u not in node_neighbors:
            node_neighbors[u] = set()
        node_neighbors[u].add(v)
        if v not in node_neighbors:
            node_neighbors[v] = set()
        node_neighbors[v].add(u)
    # 补充孤立节点（邻居数为0）
    all_nodes = set(idx) | set(edge_index.flatten())
    for node in all_nodes:
        if node not in node_neighbors:
            node_neighbors[node] = set()
        
    score_neighbor_ratio_epoch = []
    score_neighbor_ratio_in_neighbor_epoch = []
    high_attn_node_neighbor_count_epoch = []  # 单epoch下各阈值的邻居的邻居数
    all_neighbor_cnt = len(edge_set) * 2
    for k,th in tqdm(enumerate(thresh_holds),desc="neighbor"):
        score_per_n = score_matri>th
        score_cnt = np.sum(score_per_n)
        score_neighbor_cnt = 0
        high_attn_neighbor_total = 0  # 高分节点对的邻居的邻居总数
        high_attn_total = 0
        high_attn_node_pair_set = set()  # 避免重复统计同一节点对
        high_score_neighbor_ratio_in_neighbor = []
        for i in range(score_matri.shape[0]):
            high_score_neighbor_cnt = 0
            node_neighbor_cnt = 0
            for j in range(score_matri.shape[1]):
                # 获取对应的节点索引
                u = idx[i]
                v = idx[j]
                # 检查是否在边集合中（O(1)查询
                edge_tuple = (min(u, v), max(u, v))
                if i!= j:
                    if edge_tuple in edge_set:
                        node_neighbor_cnt += 1
                        # if high attn and neighbor
                        if score_per_n[i][j]:
                            score_neighbor_cnt += 1
                    # 新增：统计高分节点对的邻居的邻居数（去重）
                    if score_per_n[i][j] and edge_tuple in edge_set:
                        high_score_neighbor_cnt += 1
                    if score_per_n[i][j] and edge_tuple not in high_attn_node_pair_set: # TODO:是否需要添加uv是邻居的前提。
                        high_attn_node_pair_set.add(edge_tuple)
                        # u的邻居数 + v的邻居数
                        # neighbor_num = len(node_neighbors[u])
                        neighbor_num = len(node_neighbors[v])
                        high_attn_neighbor_total += neighbor_num
                        high_attn_total += 1
                        # high_attn_neighbor_total += (u_neighbor_num + v_neighbor_num)
            high_score_neighbor_ratio_in_neighbor.append(high_score_neighbor_cnt/node_neighbor_cnt if high_score_neighbor_cnt!=0 else 0)
        hist(high_score_neighbor_ratio_in_neighbor,"邻居中高注意力的比例","频数","邻居中高分注意力的比例",f"high_score_neighbor_ratio/per{pers[k]}",f"high_score_neighbor_ratio_epoch{epoch}.png")
        high_attn_node_neighbor_count_epoch.append(high_attn_neighbor_total/high_attn_total)
        score_neighbor_ratio_epoch.append(score_neighbor_cnt/score_cnt)
        score_neighbor_ratio_in_neighbor_epoch.append(score_neighbor_cnt/all_neighbor_cnt)
    score_neighbor_ratio_list.append(score_neighbor_ratio_epoch)
    score_neighbor_ratio_in_neighbor_list.append(score_neighbor_ratio_in_neighbor_epoch)
    high_attn_node_neighbor_neighbor_num.append(high_attn_node_neighbor_count_epoch)

def relativity(score_matri:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int):
    "节点相互高注意的比例"
    score_flat=score_matri.flatten()
    thresh_holds=np.percentile(score_flat,pers)
    score_relativity_ratio_epoch = []
    for k,th in tqdm(enumerate(thresh_holds),desc="relativity"):
        score_per_n = score_matri>th
        score_cnt = np.sum(score_per_n)
        score_relativity_cnt = 0
        for i in range(score_matri.shape[0]):
            for j in range(i+1,score_matri.shape[1]):
                if score_per_n[i][j] and score_per_n[j][i]:
                    if score_matri[i][j]//20==score_matri[j][i]//20:
                        score_relativity_cnt+=1
        score_relativity_ratio_epoch.append(score_relativity_cnt/score_cnt)
    score_relativity_ratio_list.append(score_relativity_ratio_epoch)

def high_attn_node(score_matri:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int):
    "节点被其他节点高分注意的节点比例，例如总节点数为100，一个节点被50个节点高分注意，那么其比例为50%"
    if not os.path.exists(f"./{vis_dir}/node_attned"):
        os.makedirs(f"./{vis_dir}/node_attned")
    score_flat=score_matri.flatten()
    thresh_holds=np.percentile(score_flat,pers)
    node_attned = []
    for k,th in enumerate(thresh_holds):
        node_attned_pern = []
        for i in range(score_matri.shape[0]):
            high_score_cnt = np.sum(score_matri[:, i] > th)
            node_attned_pern.append(high_score_cnt/score_matri.shape[1])
        hist(node_attned_pern,"一个节点被高注意力的比例（占总结点数）","频数","被高注意节点的高分占比比例",f"node_attned/per_{pers[k]}",f"attned_per{pers[k]}_{epoch}.png")

def distance(score_matri:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int):
    "计算高分注意力里面节点间的距离统计"
    import networkx as nx
    # 从edge_index构建图
    G = nx.Graph()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    score_flat=score_matri.flatten()
    thresh_holds=np.percentile(score_flat,pers)
    for k,th in tqdm(enumerate(thresh_holds),desc="distance"):
        score_per_n = score_matri > th
        high_attention_indices = np.argwhere(score_per_n)
        distances = []
        for i, j in high_attention_indices:
            u = idx[i]
            v = idx[j]
            try:
                # 计算节点对的最短路径长度
                dis = nx.shortest_path_length(G, source=u, target=v)
                distances.append(dis)
            except nx.NetworkXNoPath:
                distances.append(0)
            except Exception as e:
                log(f"计算节点对 ({u}, {v}) 距离时出错: {e}")
                continue
        distance_counts = Counter(distances)
        sorted_distances = sorted(distance_counts.keys())
        counts = [distance_counts[d] for d in sorted_distances]
        bar(sorted_distances,counts,"高注意节点的距离统计",f"dis/per{pers[k]}",f"distance_epoch{epoch}.jpg")

def mean_score_of_distance(score_matri:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int):
    "各个距离的注意力分数均值"
    G = nx.Graph()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    sub_nodes_num = len(idx)
    score_of_dis = {}
    score_of_dis_cnt = {}
    for i in tqdm(range(sub_nodes_num),desc="mean_score_of_distance"):
        for j in range(sub_nodes_num):
            try:
                # 计算节点对的最短路径长度
                dis = nx.shortest_path_length(G, source=idx[i], target=idx[j])
                score_of_dis_cnt[dis] = score_of_dis_cnt[dis]+1 if dis in score_of_dis else 1
                score_of_dis[dis] = score_of_dis[dis]+score_matri[i][j] if dis in score_of_dis else score_matri[i][j]
            except Exception:
                score_of_dis_cnt[0] = score_of_dis_cnt[0]+1 if dis in score_of_dis else 1
                score_of_dis[0] = score_of_dis[0]+score_matri[i][j] if 0 in score_of_dis else score_matri[i][j]
    bar([key for key,val in score_of_dis.items()],[val/score_of_dis_cnt[key] for key,val in score_of_dis.items()],"各个距离的注意力分数均值","mean_score_of_distance",f"mean_score_of_distance_epoch_{epoch}")
    if epoch==50:
        bar([key for key,val in score_of_dis_cnt.items()],[score_of_dis_cnt[key] for key,val in score_of_dis_cnt.items()],"各个距离的频数","distance",f"distance_epoch_{epoch}")

def feature_sim_high_score(score_matri:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,feature_i:np.ndarray,epoch:int):
    G = nx.Graph()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    score_flat=score_matri.flatten()
    s_pers = [90,20]
    thresh_holds=np.percentile(score_flat,s_pers)
    edge_set = set()
    for u, v in edge_index.T:
        edge_set.add((min(u, v), max(u, v)))
    # 提前计算所有相似度
    feature_norm = feature_i / (np.linalg.norm(feature_i, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = feature_norm @ feature_norm.T
    def collector(mask,x_label,y_label,title,dir,file_name):
        # 获取满足条件的索引，排除自注意力(i=j)
        rows, cols = np.where(mask)
        # 排除对角线元素（自注意力）
        non_self_mask = rows != cols
        rows = rows[non_self_mask]
        cols = cols[non_self_mask]
        scatter_score_y = [[],[]]# 0:邻居, 1:非邻居
        scatter_sim_x = [[],[]]
        for i, j in zip(rows,cols):
            u = idx[i]
            v = idx[j]
            edge_tuple = (min(u, v), max(u, v))
            if edge_tuple in edge_set:# is neighbor
                scatter_score_y[0].append(score_matri[i,j])
                scatter_sim_x[0].append(similarity_matrix[i,j])
            else:
                scatter_score_y[1].append(score_matri[i,j])
                scatter_sim_x[1].append(similarity_matrix[i,j])
        scatter(scatter_sim_x,scatter_score_y,x_label,y_label,title,dir,file_name)
    high_score_per_mask = score_matri > thresh_holds[0]
    collector(high_score_per_mask,"余弦相似度","注意力分数","大于90分位数下注意力与特征相似度关系","fea_sim",f"fea_sim_hper_90_epoch_{epoch}")
    high_score_per_mask = score_matri < thresh_holds[1]
    collector(high_score_per_mask,"余弦相似度","注意力分数","小于20分位数下注意力与特征相似度关系","fea_sim",f"fea_sim_lper_20_epoch_{epoch}")

def mask_high_attn(score_agg:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int) -> torch.Tensor:
    score_agg=score_agg.cpu().detach().numpy() if isinstance(score_agg,torch.Tensor) else score_agg
    idx = idx.cpu().detach().numpy() if isinstance(idx,torch.Tensor) else idx
    edge_index=edge_index.cpu().detach().numpy() if isinstance(edge_index,torch.Tensor) else edge_index
    if score_agg is not None:
        score_flat=score_agg.flatten()
        thresh_hold=np.percentile(score_flat,20) # 一次跑一个数据
        mask = score_agg<thresh_hold
        return torch.tensor(mask, dtype=torch.bool)
    else:
        return None

def homo_node_mask(edge_index, idx_i, mask_ratio=0.5):
    """
    极简版：用NetworkX筛选邻居，生成屏蔽掩码（True=屏蔽该边）
    """
    # 1. 转NetworkX图（无向）
    G = nx.Graph()
    G.add_edges_from(edge_index.T.cpu().numpy().tolist())
    mask = torch.zeros((len(idx_i), len(idx_i)), dtype=torch.bool)
    node_to_local_idx = {node: idx for idx, node in enumerate(idx_i.cpu().numpy())}
    for local_i,node in enumerate(idx_i.cpu().numpy()):
        # 找同质点邻居
        neighbors = [n for n in G.neighbors(node) if n in node_to_local_idx]
        # if len(neighbors) <= 1: continue
        np.random.shuffle(neighbors)
        mask_num = int(len(neighbors) * mask_ratio)
        masked_neis = neighbors[-mask_num:]
        # 标记掩码（无向图检查双向边）
        for nei in masked_neis:
            # 找到边在全局edge_index中的位置并设为True
            local_j = node_to_local_idx[nei]  # 邻居在idx_i中的局部位置
            mask[local_i, local_j] = True     # 标记屏蔽
    return mask

def vis_interface(score_matri,idx,edge_index,feature_i,epoch,args):
    if epoch % 50 == 0:
        score_matri=score_matri.cpu().detach().numpy() if isinstance(score_matri,torch.Tensor) else score_matri
        idx = idx.cpu().detach().numpy() if isinstance(idx,torch.Tensor) else idx
        edge_index=edge_index.cpu().detach().numpy() if isinstance(edge_index,torch.Tensor) else edge_index
        feature_i=feature_i.cpu().detach().numpy() if isinstance(feature_i,torch.Tensor) else feature_i
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        # high_attn(score_matri,epoch)
        # neighbor_high_attn(score_matri,edge_index,idx,epoch)
        # neighbor(score_matri,idx,edge_index,epoch)
        # relativity(score_matri,idx,edge_index,epoch)
        # high_attn_node(score_matri,idx,edge_index,epoch)
        # # distance(score_matri,idx,edge_index,epoch)
        # mean_score_of_distance(score_matri,idx,edge_index,epoch)
        feature_sim_high_score(score_matri,idx,edge_index,feature_i,epoch)
        epochs.append(epoch)
    if epoch == args.epochs-1:
        # pics_to_gif(score_hist_flist,"./vis/score_var.gif")
        labels = [f"{p} %分位数" for p in pers]
        # plot(epochs,np.array(score_neighbor_ratio_list).T,labels,vis_dir,"高注意力邻居占比","高注意力邻居占比.png")
        # plot(epochs,np.array(high_attn_node_neighbor_neighbor_num).T,labels,vis_dir,"被高注意节点邻居数均值","被高注意节点邻居数均值.png")
        # plot(epochs,np.array(score_neighbor_ratio_in_neighbor_list).T,labels,vis_dir,"邻居中高注意力邻居占比","邻居中高注意力邻居占比.png")
        # plot(epochs,np.array(score_relativity_ratio_list).T,labels,vis_dir,"高注意力相对性占比","高注意力相对性占比.png")
        plot(epochs,[train_acc,test_acc,val_acc],["train_acc","test_acc","val_acc"],vis_dir,"准确率","acc_epochs.png")
