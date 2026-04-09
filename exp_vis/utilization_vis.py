import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

def multi_plot(plot_params, my_params, figpath=None):
    pylab.rcParams.update(plot_params)
    plt.rcParams['pdf.fonttype'] = 42

    n_rows, n_cols = my_params.get('axes', [1, 1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12,5), constrained_layout=True)
    
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    axes_params = my_params.get('axes_params', [])

    for i, (ax, data) in enumerate(zip(axes, axes_params)):
        group_names = list(data['y_val'].keys())
        xticks = data['x_ticks']
        titles = data['title']
        y_vals = data['y_val']
        ind = np.arange(len(xticks))
        
        handles = []
        use_twin = data.get('use_twin', False)
        ax_secondary = ax.twinx() if use_twin else None

        for g_idx, g_name in enumerate(group_names):
            # 将数据转为 np.array，并将 0 替换为 None/NaN 以后便折线断开
            raw_data = np.array(y_vals[g_name], dtype=float)
            plot_data = raw_data.copy() # 保留原始数据用于 OOM 标记
            # plot_data = np.where(raw_data == 0, np.nan, raw_data)
            
            target_ax = ax_secondary if (use_twin and g_idx > 0) else ax
            
            # 绘制折线
            print(f"绘制 {g_name} 数据: {y_vals[g_name]} on {'secondary' if (use_twin and g_idx > 0) else 'primary'} axis, positions: {ind}")
            line, = target_ax.plot(ind, plot_data, 
                                   color=data['colors'][g_idx],
                                #    marker=data['markers'][g_idx], 
                                   markersize=8,
                                   linewidth=2,
                                   label=g_name, 
                                   alpha=0.8)
            handles.append(line)

            # # --- OOM 标记逻辑 ---
            # for j, val in enumerate(raw_data):
            #     if val == 0:
            #         # 在折线中断处标记红色 X 或 OOM 字样
            #         target_ax.text(ind[j], 0.1, xticks[j]+' OOM', 
            #                        ha='center', va='bottom', 
            #                        fontsize=10, color='red',
            #                        fontweight='bold', transform=target_ax.get_xaxis_transform())
            
            # 针对特定轴的设置
            if use_twin and g_idx > 0:
                target_ax.set_ylabel(data.get('y2_label', g_name),fontweight=plot_params['font.weight'])
                if 'y2_lim' in data: target_ax.set_ylim(*data['y2_lim'])
                if data.get('y2_log', False): ax.set_yscale('log',base=2)
            else:
                ax.set_ylabel(data.get('y1_label', 'Value'), fontweight=plot_params['font.weight'])
                if 'y1_lim' in data: ax.set_ylim(*data['y1_lim'])
                if data.get('y1_log', False): ax.set_yscale('log',base=2)

        # 公共设置
        ax.set_xticks(ind[::100]) # 只显示部分 xticks，避免过密
        ax.set_xticklabels([xticks[i]/10 for i in range(0, len(xticks), 100)], fontweight=plot_params['font.weight'])
        ax.set_xlabel('Time (s)', fontweight=plot_params['font.weight'])
        ax.set_title(titles,y=-0.25, fontweight=plot_params['font.weight'])
        ax.grid(True, linestyle=':', alpha=0.6)

        # 合并图例
        handles, labels = axes[0].get_legend_handles_labels()
        # 在 figure 级别添加图例
        # 修改这一段
        fig.legend(handles, labels,
                loc='upper center', 
                bbox_to_anchor=(0.5, 1.15), # 稍微调高一点，防止压到标题
                ncol=3, 
                frameon=False, 
                prop={'size': plot_params['legend.fontsize'], 'weight': plot_params['font.weight']}) # 使用 prop 字典强制控制

    if figpath:
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.show()
    
def load_csv_data(file_path, max_seconds=1000):
    """
    读取CSV，提取第二列（利用率），并截取前100秒
    """
    import pandas as pd
    try:
        # 假设没有表头，如果有表头请去掉 header=None
        df = pd.read_csv(file_path, header=None, names=['timestamp', 'util', 'mem'])
        # 提取利用率，取前 100 行（代表 100 秒）
        util_data = df['util'].values[:max_seconds].tolist()
        print(f"mean utilization from {file_path}: {np.mean(util_data):.2f}%")
        return util_data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return [0] * max_seconds

if __name__ == "__main__":
    params = {
        'font.family': 'Arial',
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.titlesize': 20,
        'font.weight': 'bold',
    }
    
    # 配置参数：通过 use_twin 开关控制
    my_params = {
        'axes': [1, 1],
        'bar_width': 0.25,
        'axes_params' : [
            {
                'y_val' : {
                    'TorchGT': load_csv_data('TorchGT.log'),
                    'UnifiedGT': load_csv_data('UnifiedGT.log'),
                    'NeutronGT': load_csv_data('NeutronGT.log'),
                },
                'x_ticks' : np.arange(0, 1000).tolist(),
                'title' : '',
                'y1_lim' : (-5, 105),
                # 'y2_lim' : (1e-5, 10),
                'y1_log' : False,
                # 'y2_log' : True,
                'y1_label' : 'Utilization (%)',
                # 'y2_label' : 'Capture Rate',
                # 'markers' : ['o', 's', '^'],
                'colors' : ['#1f77b4', '#ff7f0e','#73C991'],
                # 'use_twin': True,              # 是否开启双 Y 轴
                'legend_loc':'upper left'
            }
        ]
    }

    multi_plot(params, my_params, figpath='gpu_utilization.pdf')
