#!/bin/bash

# ==============================================================================
# 1. 动态 GPU 节点数量解析
# ==============================================================================
# 接收第一个参数作为可见的 GPU 列表，例如: bash run_all_experiments.sh 0,1,2,3
DEVICES=$1

if [ -z "$DEVICES" ]; then
    echo "❌ 错误: 请传入 CUDA_VISIBLE_DEVICES 作为参数！"
    echo "💡 示例: bash $0 0,1,2,3"
    exit 1
fi

# 通过统计逗号的数量来计算使用的 GPU 个数
device_num=$(echo $DEVICES | tr -cd ',' | wc -c)
device_num=$((device_num + 1))

echo "================================================================="
echo "🖥️  系统配置: 检测到 $device_num 张 GPU (可用设备: $DEVICES)"
echo "================================================================="

CURRENT_DATE=$(date +"%Y%m%d_%H%M")
mkdir -p experiment_logs

# "ogbn-papers100M"
DATASETS=( "ogbn-products")

# 模型配置：格式为 "日志代号|传入的--model名|n_layers|hidden_dim|num_heads"
# 注: 根据你的测试脚本，ffn_dim 通常与 hidden_dim 保持一致，脚本中会自动关联
MODELS=(
    # "Graphormer-Slim|graphormer|4|64|8" 测完了
    "Graphormer-Large|graphormer|12|768|32"
    "GT|gt|4|128|8"
)

# 运行策略配置：格式为 "日志代号|attn_type|是否使用reorder(true/false)"
ATTENTIONS=(
    "Full_NoReorder|full|false"
    "Flash_NoReorder|flash|false"
    "Sparse_Reorder|sparse|true"
)

# 固定的训练超参数
SEQ_LEN=64000
EPOCHS=1000
MASTER_PORT=8081


for dataset in "${DATASETS[@]}"; do
    for model_info in "${MODELS[@]}"; do
        
        # 使用 IFS 解析模型参数
        IFS='|' read -r model_alias model_arg n_layers hidden_dim num_heads <<< "$model_info"
        ffn_dim=$hidden_dim  # 保持 ffn_dim 与 hidden_dim 一致

        for attn_info in "${ATTENTIONS[@]}"; do
            
            # 使用 IFS 解析注意力策略参数
            IFS='|' read -r strategy_alias attn_type use_reorder <<< "$attn_info"

            # 动态生成日志文件名
            LOG_FILE="experiment_logs/${dataset}_${model_alias}_${strategy_alias}_${CURRENT_DATE}.log"

            # 动态处理 --reorder 开关标志
            REORDER_FLAG=""
            if [ "$use_reorder" = "true" ]; then
                REORDER_FLAG="--reorder"
            fi

            echo "🚀 正在启动: [数据集: $dataset] | [模型: $model_alias] | [策略: $strategy_alias]"
            echo "📝 实时日志: $LOG_FILE"

            # 组装并执行 torchrun 命令
            CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
                --nproc_per_node=${device_num} \
                --master_port ${MASTER_PORT} \
                main_sp_node_level.py \
                --dataset ${dataset} \
                --seq_len ${SEQ_LEN} \
                --n_layers ${n_layers} \
                --hidden_dim ${hidden_dim} \
                --ffn_dim ${ffn_dim} \
                --num_heads ${num_heads} \
                --epochs ${EPOCHS} \
                --model ${model_arg} \
                --distributed-backend 'nccl' \
                --attn_type ${attn_type} \
                ${REORDER_FLAG} \
                > "${LOG_FILE}" 2>&1

            # 捕获退出状态码，防止某个 OOM 导致整个实验矩阵崩溃退出
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                echo "✅ 运行成功！"
            else
                echo "❌ 运行失败 (退出码: $EXIT_CODE)，可能发生 OOM。跳过并继续下一组..."
            fi
            echo "-----------------------------------------------------------------"
            
            # 短暂休眠，确保系统的分布式通信端口（如 8081）彻底释放，防止下一次启动报 Address already in use
            sleep 3

        done
    done
done

echo "所有实验评测已完毕！请检查 experiment_logs 目录获取详细指标。"