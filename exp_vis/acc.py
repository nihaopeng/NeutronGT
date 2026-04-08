import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

def parallel_bar(plot_params, my_params, figpath=None):
    pylab.rcParams.update(plot_params)
    plt.rcParams['pdf.fonttype'] = 42

    # 1. 基础参数解析
    n_rows, n_cols = my_params.get('axes', [1, 1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=my_params.get('figsize', (6 * n_cols, 4 * n_rows)), constrained_layout=True)
    
    # 兼容单子图情况（确保 axes 是列表）
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    axes_params = my_params.get('axes_params', [])
    bar_width = my_params.get('bar_width', 0.3)

    for i, (ax, data) in enumerate(zip(axes, axes_params)):
        group_names = list(data['y_val'].keys())
        xticks = data['x_ticks']
        titles = data['title']
        y_vals = data['y_val']
        n_groups = len(group_names)

        ind = np.arange(len(xticks))
        
        # 计算起始偏移量，使多根柱子整体居中
        # 比如 2 根柱子，偏移分别是 -width/2, +width/2
        offset_start = (n_groups - 1) * bar_width / 2
        
        handles = [] # 用于存储图例句柄
        
        # 检查是否需要开启双轴
        use_twin = data.get('use_twin', False)
        ax_secondary = ax.twinx() if use_twin else None

        for g_idx, g_name in enumerate(group_names):
            # 计算当前组柱子的 X 位置
            pos = ind - offset_start + g_idx * bar_width
            # 选择坐标轴：如果是双轴且是第二组数据，画在右轴
            target_ax = ax_secondary if (use_twin and g_idx > 0) else ax
            # 绘图
            print(f"绘制 {g_name} 数据: {y_vals[g_name]} on {'secondary' if (use_twin and g_idx > 0) else 'primary'} axis, positions: {pos}")
            bar = target_ax.bar(pos, y_vals[g_name], bar_width, 
                                color=data['colors'][g_idx],
                                hatch=data['hatchs'][g_idx],
                                label=g_name, edgecolor='black', alpha=0.7)
            
            handles.append(bar)
            
            # --- 新增：OOM 标记逻辑 ---
            for j, val in enumerate(y_vals[g_name]):
                if val == 0: # 或者使用 np.isnan(val)
                    # 在柱子底部或稍微向上的位置写上 "OOM"
                    target_ax.text(pos[j], 0.01, 'OOM', 
                                   ha='center', va='bottom', 
                                   fontsize=14, color='red', 
                                   fontweight='bold', rotation=90)
            # -----------------------

            # 针对特定轴的个性化设置
            if use_twin and g_idx > 0:
                y2_label = data.get('y2_label', None)
                if y2_label is not None:
                    target_ax.set_ylabel(data.get('y2_label', g_name),fontweight=plot_params['font.weight'])
                if 'y2_lim' in data: target_ax.set_ylim(*data['y2_lim'])
                if data.get('y2_log', False): target_ax.set_yscale('log')
            else:
                y1_label = data.get('y1_label', None)
                if y1_label is not None:
                    ax.set_ylabel(data.get('y1_label', g_name),fontweight=plot_params['font.weight'])
                if 'y1_lim' in data: ax.set_ylim(*data['y1_lim'])
                if data.get('y1_log', False): ax.set_yscale('log')

        # 公共设置
        ax.set_xticks(ind)
        ax.set_xticklabels(xticks)
        ax.set_title(titles,y=-0.25,fontweight=plot_params['font.weight'])
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # 合并图例（处理 twinx 导致的图例分离问题）
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 1.10), # 稍微调高一点，防止压到标题
                ncol=3, 
                frameon=False, 
                prop={'size': plot_params['legend.fontsize'], 'weight': plot_params['font.weight']}) # 使用 prop 字典强制控制

    if figpath:
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    params = {
        'font.family': 'Arial',
        'axes.labelsize': 15,
        'xtick.labelsize': 18,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'axes.titlesize': 16,
        'font.weight': 'bold'
    }
    
    # 配置参数：通过 use_twin 开关控制
    my_params = {
        'axes': [1, 3],
        'figsize' : (9, 5),
        'bar_width': 0.25,
        'axes_params' : [
            {
                'y_val' : {
                    'TorchGT': [13.07, 15, 27.33, 25.86],
                    # 'UnifiedGT':[52.259,0,74.232,0],
                    'NeutronGT': [12.80741, 12.29618, 29.39479, 30.30985],
                },
                'x_ticks' : ['OVA','RDT','OPT','AZ'],
                'title' : '(a) GT',
                'y1_lim' : (0, 100),
                # 'y2_lim' : (1e-5, 10),
                'y1_log' : False,
                # 'y2_log' : True,
                'y1_label' : 'Test Accuracy (%)',
                # 'y2_label' : 'Capture Rate',
                'hatchs' : ['///', '...','\\\\\\'],
                'colors' : ['#1f77b4', '#ff7f0e','#ffff0e'],
                # 'use_twin': True,              # 是否开启双 Y 轴
                'legend_loc':'upper left'
            },
            {
                'y_val' : {
                    'torchGT': [16.18, 40.63, 38.97, 26.72],
                    # 'UnifiedGT':[47.241,87.996,74.151,0],
                    'NeutronGT': [27.35271, 46.95454, 47.24095, 59.05403],
                },
                'x_ticks' : ['OVA','RDT','OPT','AZ'],
                'title' : '(b) $GPH_{Slim}$',
                'y1_lim' : (0, 100),
                # 'y2_lim' : (1e-5, 10),
                'y1_log' : False,
                # 'y2_log' : True,
                # 'y1_label' : 'Test Accuracy (%)',
                # 'y2_label' : 'Capture Rate',
                'hatchs' : ['///', '...','\\\\\\'],
                'colors' : ['#1f77b4', '#ff7f0e','#ffff0e'],
                # 'use_twin': True,              # 是否开启双 Y 轴
                'legend_loc':'upper left'
            },
            {
                'y_val' : {
                    'torchGT': [45.56, 70, 62.57, 64.81],
                    # 'UnifiedGT':[0,0,0,0],
                    'NeutronGT': [50.08166, 91.35715, 80.9042, 80.00465],
                },
                'x_ticks' : ['OVA','RDT','OPT','AZ'],
                'title' : '(c) $GPH_{Large}$',
                'y1_lim' : (0, 100),
                # 'y2_lim' : (1e-5, 10),
                'y1_log' : False,
                # 'y2_log' : True,
                # 'y1_label' : 'Test Accuracy (%)',
                # 'y2_label' : 'Capture Rate',
                'hatchs' : ['///', '...','\\\\\\'],
                'colors' : ['#1f77b4', '#ff7f0e','#ffff0e'],
                # 'use_twin': True,              # 是否开启双 Y 轴
                'legend_loc':'upper left'
            }
        ]
    }

    parallel_bar(params, my_params, figpath='acc_20_epoch_bar.pdf')
