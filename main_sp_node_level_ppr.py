import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from core.metisPartition import weightMetis_keepParent
from models.graphormer_dist_node_level import Graphormer
from models.gt_dist_node_level import GT
from models.gt_dist_node_level_single_window import GT_SW
from utils.lr import PolynomialDecayLR
import argparse
import os
import time
import random
import pandas as pd
import torch.distributed as dist
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    set_last_batch_global_token_indices,
)
from gt_sp.reducer import sync_params_and_buffers, Reducer
from gt_sp.evaluate import calc_acc
from gt_sp.utils import LossStagnationDetector, compute_graphormer_spatial_pos_only, get_node_degrees, random_split_idx
from utils.parser_node_level import parser_add_main_args
from core.pprPartition import add_isolated_connections, personal_pagerank,build_adj_fromat
from utils.vis import vis_interface
import utils.vis as vis
import utils.logger as logger

def build_graph_struct_info(args,N,edge_index,feature,world_size,topk=50,n_parts=50,related_nodes_topk_rate=5,connect_prob=0.01):
    # --------------------计算结构信息------------------------------------------------------------
    # =================== ppr partition =========================
    partitioned_results = []
    # if args.rank == 0:
    sorted_ppr_matrix = personal_pagerank(edge_index,0.85,topk=topk)
    sorted_ppr_matrix = add_isolated_connections(sorted_ppr_matrix,N,connect_prob=connect_prob)
    csr_adjacency,eweights,adj_weight = build_adj_fromat(sorted_ppr_matrix=sorted_ppr_matrix)
    wm = weightMetis_keepParent(
        csr_adjacency=csr_adjacency, 
        eweights=eweights,
        n_parts=n_parts,
        feature=feature,
        edge_index=edge_index,
        related_nodes_topk_rate=related_nodes_topk_rate,
        attn_type=args.attn_type,
        sorted_ppr_matrix=sorted_ppr_matrix)
    # partitioned_results = wm.partitioned_results
    # sub_edge_index_list = wm.sub_edge_index_for_partitioned_results
    # duplicated_nodes = wm.duplicated_nodes
    # partitioned_results = metis_partition(csr_adjacency,eweights,n_parts=n_parts)
    # partitioned_results = ppr_partition(sorted_ppr_matrix,flatten_train_idx.cpu().numpy(),num_set=100)
    
    # ===== 计算centrality encoding =====
    graph_in_degree, graph_out_degree = None, None
    if args.struct_enc == "True":
        graph_in_degree, graph_out_degree = get_node_degrees(edge_index, N)
    # ==================================

    # ---------------------------------------------------------
    print("node len:",end="")
    sum_nodes_in_compute = 0 
    for p in partitioned_results:
        print(len(p),end="|")
        sum_nodes_in_compute += len(p)
    print("\nedge len:",end="")
    sum_edges_in_compute = 0 
    for p in wm.sub_edge_index_for_partition_results:
        print(len(p[0]),end="|")
        sum_edges_in_compute += len(p[0])
    print(f"\nsum nodes in compute:{sum_nodes_in_compute},sum edges in compute:{sum_edges_in_compute}")
    
    return graph_in_degree,graph_out_degree,sorted_ppr_matrix,wm

def build_model(args,feature,device,y,**kwargs):
    graph_in_degree = kwargs["graph_in_degree"]
    graph_out_degree = kwargs["graph_out_degree"]
    if args.model == "graphormer":
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=feature.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=y.max().item()+1,
            attn_bias_dim=args.attn_bias_dim,
            dropout_rate=args.dropout_rate,
            input_dropout_rate=args.input_dropout_rate,
            attention_dropout_rate=args.attention_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node
        ).to(device)
    elif args.model == "gt":
        model = GT(
           n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=feature.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=y.max().item()+1,
            attn_bias_dim=args.attn_bias_dim,
            dropout_rate=args.dropout_rate,
            input_dropout_rate=args.input_dropout_rate,
            attention_dropout_rate=args.attention_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node,
            args=args,
            num_in_degree = torch.max(graph_in_degree).item() if graph_in_degree is not None else 0,
            num_out_degree = torch.max(graph_out_degree).item() if graph_out_degree is not None else 0,
            num_spatial=args.max_dist+2,
            num_edges=args.max_num_edges,
            max_dist=args.max_dist,
            edge_dim=64
        ).to(device)
    elif args.model == "gt_sw": # only use in ppr
        model = GT_SW(
           n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=feature.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=y.max().item()+1,
            attn_bias_dim=args.attn_bias_dim,
            dropout_rate=args.dropout_rate,
            input_dropout_rate=args.input_dropout_rate,
            attention_dropout_rate=args.attention_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node,
            args=args,
            num_in_degree = torch.max(graph_in_degree).item() if graph_in_degree is not None else 0,
            num_out_degree = torch.max(graph_out_degree).item() if graph_out_degree is not None else 0,
            num_spatial=args.max_dist+2,
            num_edges=args.max_num_edges,
            max_dist=args.max_dist,
            edge_dim=64
        ).to(device)
    return model

@torch.no_grad()
def eval_epoch(args,model,partitions,feature,y,split_idx,device,**kwargs):
    # ---verify that whether the param was changed---
    # curr_param = None
    # for param in model.parameters():
    #     if param.requires_grad:
    #         curr_param = param
    #         break
    # print(f"{curr_param}")
    # -----------------------------------------------

    graph_in_degree = kwargs["graph_in_degree"]
    graph_out_degree = kwargs["graph_out_degree"]
    spatial_pos_list = kwargs["spatial_pos_list"]
    epoch = kwargs["epoch"]
    sub_edge_index_list = kwargs["sub_edge_index_list"]
    model.eval()
    y_train_true,y_valid_true,y_test_true = [], [], []
    y_train_pred,y_valid_pred,y_test_pred = [], [], []
    
    for i,idx in enumerate(partitions):
        x_i = feature[idx].to(device)
        attn_bias,in_degree,out_degree = None,None,None
        edge_index_i,edge_input_i,mask,spatial_pos_i = sub_edge_index_list[i].to(device),None,None,None
        if args.struct_enc=="True":
            # == node degree in the sequnence ==
            in_degree = graph_in_degree[idx].to(device)
            out_degree = graph_out_degree[idx].to(device)
            spatial_pos_i = spatial_pos_list[i].to(device)
        out_i,_,_ = model(x_i, attn_bias, edge_index_i,in_degree,out_degree, spatial_pos_i,edge_input_i,attn_type=args.attn_type,mask=mask)
        # print(f"out i:{out_i}")
        mask_train = torch.isin(idx.to(device), split_idx["train"].to(device)).to('cpu')
        mask_valid = torch.isin(idx.to(device), split_idx["valid"].to(device)).to('cpu')
        mask_test = torch.isin(idx.to(device), split_idx["test"].to(device)).to('cpu')
        y_train_true.append(y[idx][mask_train])
        y_valid_true.append(y[idx][mask_valid])
        y_test_true.append(y[idx][mask_test])
        y_train_pred.append(out_i.argmax(1)[mask_train])
        y_valid_pred.append(out_i.argmax(1)[mask_valid])
        y_test_pred.append(out_i.argmax(1)[mask_test])
        # if args.rank == 0 and i == 0:
        #     print(f"y test pred:{y_test_pred}")
    y_train_true = torch.cat(y_train_true)
    y_valid_true = torch.cat(y_valid_true)
    y_test_true = torch.cat(y_test_true)
    y_train_pred = torch.cat(y_train_pred)
    y_valid_pred = torch.cat(y_valid_pred)
    y_test_pred = torch.cat(y_test_pred)
    train_acc = calc_acc(y_train_true,y_train_pred)
    valid_acc = calc_acc(y_valid_true,y_valid_pred)
    test_acc = calc_acc(y_test_true,y_test_pred)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Epoch: {:03d}, train_acc: {:.5f}%, valid_acc: {:.5f}%, test_acc: {:.5f}%,".format(epoch, train_acc*100, valid_acc*100, test_acc*100))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return train_acc*100, valid_acc*100, test_acc*100

def train_epoch(args,model:torch.nn.Module,partitions,feature,y,optimizer,lr_scheduler,world_size,split_idx,device,**kwargs):
    graph_in_degree = kwargs["graph_in_degree"]
    graph_out_degree = kwargs["graph_out_degree"]
    spatial_pos_list = kwargs["spatial_pos_list"]
    epoch = kwargs["epoch"]
    sub_edge_index_list = kwargs["sub_edge_index_list"]
    duplicated_nodes = kwargs["duplicated_nodes"]
    model.train()
    loss_list, iter_t_list,iter_cpu2gpu_t_list,epoch_t_list,epoch_cpu2gpu_t_list = [], [], [], [], []
    
    scores = []
    for i,idx in enumerate(partitions):
        idx_i = idx
        t0 = time.time()
        x_i = feature[idx_i].to(device)
        attn_bias,in_degree,out_degree = None,None,None
        edge_index_i,edge_input_i,mask,spatial_pos_i = sub_edge_index_list[i].to(device),None,None,None
        if args.struct_enc=="True":
            # == node degree in the sequnence ==
            in_degree = graph_in_degree[idx_i].to(device)
            out_degree = graph_out_degree[idx_i].to(device)
            spatial_pos_i = spatial_pos_list[i].to(device)
        t1 = time.time()
        out_i,score_agg,score_spe = model(
            x_i, 
            attn_bias, 
            edge_index_i,
            in_degree,
            out_degree, 
            spatial_pos_i,
            edge_input_i,
            attn_type=args.attn_type,
            mask=mask,
            duplicated_nodes=duplicated_nodes
        )
        # 取最后一层
        scores.append(score_spe[args.n_layers-1])
        loss = F.nll_loss(out_i, y[idx].to(device).long(),reduction='none')
        optimizer.zero_grad(set_to_none=True)
        mask_train = torch.isin(idx.to(device), split_idx["train"].to(device))
        # print(f"mask sum:{mask_train.sum().item()}")
        if mask_train.any():
            loss = loss[mask_train].mean()
        else:
            print(f"rank:{args.rank},epoch:{epoch},no train nodes!")
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss.backward()
        # Sync all-reduce gradient
        if world_size > 1:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param.grad.div_(get_sequence_parallel_world_size())
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=get_sequence_parallel_group())
        optimizer.step()
        t2 = time.time()
        iter_t_list.append(t2 - t1)
        iter_cpu2gpu_t_list.append(t1-t0)
        # if i==0:
        #     vis_interface(score_spe[3].squeeze(0)[0],idx_i,edge_index,x_i,epoch,args)
        loss_list.append(loss.item())
    lr_scheduler.step()
    if args.rank == 0:
        epoch_t_list.append(np.sum(iter_t_list))
        epoch_cpu2gpu_t_list.append(np.sum(iter_cpu2gpu_t_list))
        print("------------------------------------------------------------------------------------")
        print("Epoch: {:03d}, Loss: {:.4f}, Epoch Time: {:.3f}s, Trans Time: {:.3f}s".format(epoch, np.mean(loss_list), np.mean(epoch_t_list),np.mean(epoch_cpu2gpu_t_list)))
        print("------------------------------------------------------------------------------------")
    return np.mean(loss_list),scores

def main():
    logger.IS_LOGGING = False
    parser = argparse.ArgumentParser(description='TorchGT node-level training arguments.')
    parser_add_main_args(parser)
    args = parser.parse_args()

    vis.vis_dir = args.vis_dir
   
    # Initialize distributed 
    initialize_distributed(args)
    device = f'cuda:{torch.cuda.current_device()}' 
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)
    
    # Dataset 
    feature = torch.load(args.dataset_dir + args.dataset + '/x.pt') # [N, x_dim]
    y = torch.load(args.dataset_dir + args.dataset + '/y.pt') # [N]
    edge_index = torch.load(args.dataset_dir + args.dataset + '/edge_index.pt') # [2, num_edges]
    N = feature.shape[0]

    if args.dataset == 'pokec':
        y = torch.clamp(y, min=0) 
    split_idx = random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)

    if args.rank == 0:
        print(args)
        print('Dataset load successfully')
        print(f"Train nodes: {split_idx['train'].shape[0]}, Val nodes: {split_idx['valid'].shape[0]}, Test nodes: {split_idx['test'].shape[0]}") 
        print(f"Training iters: {split_idx['train'].size(0) // args.seq_len + 1}, Val iters: {split_idx['valid'].size(0) // args.seq_len + 1}, Test iters: {split_idx['test'].size(0) // args.seq_len + 1}")
    
    # Broadcast train indexes to all ranks 
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    # if seq_parallel_world_size > 1:
    #     src_rank = get_sequence_parallel_src_rank()
    #     group = get_sequence_parallel_group()

    train_idx = split_idx['train']
    
    if args.rank == 0:
        flatten_train_idx = train_idx.to('cuda')
    else:
        total_numel = train_idx.numel()
        flatten_train_idx = torch.empty(total_numel,
                                device=device,
                                dtype=torch.int64)
    # Broadcast
    # if seq_parallel_world_size > 1:
    #     dist.broadcast(flatten_train_idx, src_rank, group=group)

    # Initialize global token indices
    seq_len_per_rank = get_sequence_length_per_rank()
    sub_real_seq_len = seq_len_per_rank + args.num_global_node
    global_token_indices = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))

    # Last batch fix sequence length
    if flatten_train_idx.shape[0] % args.seq_len != 0:
        last_batch_node_num = flatten_train_idx.shape[0] % args.seq_len
        if last_batch_node_num % seq_parallel_world_size != 0:
            div = last_batch_node_num // seq_parallel_world_size
            last_batch_node_num = div * seq_parallel_world_size + (seq_parallel_world_size - 1)

        x_dummy_list = [t for t in torch.tensor_split(
            torch.zeros(last_batch_node_num, ), seq_parallel_world_size, dim=0)]
        sub_split_seq_lens = [t.shape[0] for t in x_dummy_list] # e.g., [14, 14, 14, 13]
        sub_real_seq_len = max(sub_split_seq_lens) + args.num_global_node
        global_token_indices_last_batch = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))
    else:
        sub_split_seq_lens = None
        global_token_indices_last_batch = None
    set_global_token_indices(global_token_indices)
    set_last_batch_global_token_indices(global_token_indices_last_batch)

    graph_in_degree,graph_out_degree,sorted_ppr_matrix,wm \
        = build_graph_struct_info(
            args,N,edge_index,feature,seq_parallel_world_size,
            topk=50,
            n_parts=60,
            related_nodes_topk_rate=5
        )
    wm:weightMetis_keepParent = wm

    model = build_model(args,feature,device,y,graph_in_degree=graph_in_degree,graph_out_degree=graph_out_degree)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_updates,
            tot=args.epochs,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)
    
    if args.rank == 0:
        print('Model params:', sum(p.numel() for p in model.parameters()))

    # Sync params and buffers. Ensures all rank models start off at the same value
    if seq_parallel_world_size > 1:
        sync_params_and_buffers(model)

    # ===== 提前获取各设备idx ======
    partitions = []
    # for i in range(0,len(partitioned_results)): # 全都计算，一般使用gt模型，以支持注意力交换。
    for i in range(args.rank,len(wm.partitioned_results),seq_parallel_world_size):
        partitions.append(wm.partitioned_results[i])
    spatial_pos_list = None
    if args.struct_enc=="True":
        spatial_pos_list,_ = compute_graphormer_spatial_pos_only(sorted_ppr_matrix,partitions,N,max_dist=args.max_dist)
    loss_mean_list = []
    detector = LossStagnationDetector(cooldown=0)
    for epoch in range(0, args.epochs):
        sub_edge_index_list = wm.sub_edge_index_for_partition_results
        loss_mean,scores_list = train_epoch(args,model,partitions,feature,y,optimizer,lr_scheduler,seq_parallel_world_size,split_idx,device,
                    graph_in_degree=graph_in_degree,
                    graph_out_degree=graph_out_degree,
                    spatial_pos_list=spatial_pos_list,
                    epoch=epoch,
                    sub_edge_index_list=sub_edge_index_list,
                    duplicated_nodes=None)
        if args.rank == 0 and epoch % 20 == 0:
            print(f"epoch {epoch}: lr = {lr_scheduler.get_last_lr()[0]:.2e}")
            train_acc,valid_test,test_acc = eval_epoch(args,model,partitions,feature,y,split_idx,device,
                    graph_in_degree=graph_in_degree,
                    graph_out_degree=graph_out_degree,
                    spatial_pos_list=spatial_pos_list,
                    epoch=epoch,
                    sub_edge_index_list=sub_edge_index_list,
                    duplicated_nodes=None)
        
        # 窗口调整
        # =================================================================
        # if (epoch+1) % 20 == 0:
        #     partitionTree.dynamic_window_build(scores,metis_partition_nodes,remove_ratio=0.05)
        loss_mean_list.append(loss_mean)
        # if True:
        if detector(loss_mean_list):
            print("!node in and out!")
            wm.node_out(scores_list,remove_ratio=0.3)
            wm.node_in()
            partitions = []
            # for i in range(0,len(partitioned_results)): # 全都计算，一般使用gt模型，以支持注意力交换。
            for i in range(args.rank,len(wm.partitioned_results),seq_parallel_world_size):
                partitions.append(wm.partitioned_results[i])
            spatial_pos_list = None
            if args.struct_enc=="True":
                spatial_pos_list,_ = compute_graphormer_spatial_pos_only(sorted_ppr_matrix,partitions,N,max_dist=args.max_dist)
        # =================================================================   

if __name__ == "__main__":
    main()
