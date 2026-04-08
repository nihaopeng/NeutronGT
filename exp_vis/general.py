import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

def parallel_bar(plot_params, my_params, figpath=None):
    pylab.rcParams.update(plot_params)
    plt.rcParams['pdf.fonttype'] = 42

    # 1. 基础参数解析
    n_rows, n_cols = my_params.get('axes', [1, 1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), constrained_layout=True)
    
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
                    target_ax.text(pos[j], 1, 'OOM', 
                                   ha='center', va='bottom', 
                                   fontsize=10, color='red', 
                                   fontweight='bold', rotation=90)
            # -----------------------

            # 针对特定轴的个性化设置
            if use_twin and g_idx > 0:
                target_ax.set_ylabel(data.get('y2_label', g_name))
                if 'y2_lim' in data: target_ax.set_ylim(*data['y2_lim'])
                if data.get('y2_log', False): target_ax.set_yscale('log')
            else:
                ax.set_ylabel(data.get('y1_label', g_name))
                if 'y1_lim' in data: ax.set_ylim(*data['y1_lim'])
                if data.get('y1_log', False): ax.set_yscale('log')

        # 公共设置
        ax.set_xticks(ind)
        ax.set_xticklabels(xticks)
        ax.set_title(titles,y=-0.25)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # 合并图例（处理 twinx 导致的图例分离问题）
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc=data['legend_loc'], frameon=False)

    if figpath:
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.show()
    
def stack_bar(plot_params, my_params, figpath=None):
    pylab.rcParams.update(plot_params)
    plt.rcParams['pdf.fonttype'] = 42

    # 1. 基础参数解析
    n_rows, n_cols = my_params.get('axes', [1, 1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), constrained_layout=True)
    
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
        ind = np.arange(len(xticks))
        # 计算起始偏移量，使多根柱子整体居中
        # 比如 2 根柱子，偏移分别是 -width/2, +width/2
        handles = [] # 用于存储图例句柄
        # 检查是否需要开启双轴

        bottom = [0 for i in range(len(xticks))]
        for g_idx, g_name in enumerate(group_names):
            current_y = np.array(y_vals[g_name])
            target_ax = ax
            
            # 2. 绘制柱子，x 坐标使用 ind，传入当前的 bottom
            bar = target_ax.bar(ind, current_y, bar_width,
                                color=data['colors'][g_idx],
                                hatch=data['hatchs'][g_idx],
                                label=g_name, edgecolor='black', alpha=0.7,
                                bottom=bottom) # 这里的 bottom 很关键
            handles.append(bar)
            # 3. OOM 标记逻辑
            for j, val in enumerate(current_y):
                if val == 0:
                    # 标注在当前堆叠高度的上方一点点
                    text_y = bottom[j] + (max(bottom) * 0.05 if max(bottom) > 0 else 1)
                    target_ax.text(ind[j], text_y, 'OOM', 
                                   ha='center', va='bottom', 
                                   fontsize=10, color='red', 
                                   fontweight='bold', rotation=90)

            # 4. 重要：更新 bottom，为下一层叠加做准备
            bottom += current_y
            # -----------------------
            # 针对特定轴的个性化设置
            ax.set_ylabel(data.get('y1_label', g_name))
            if 'y1_lim' in data: ax.set_ylim(*data['y1_lim'])
            if data.get('y1_log', False): ax.set_yscale('log')
        # 公共设置
        ax.set_xticks(ind)
        ax.set_xticklabels(xticks,rotation=45)
        ax.set_title(titles)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        # 合并图例（处理 twinx 导致的图例分离问题）
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc=data['legend_loc'], frameon=False)

    if figpath:
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.show()

def multi_plot(plot_params, my_params, figpath=None):
    pylab.rcParams.update(plot_params)
    plt.rcParams['pdf.fonttype'] = 42

    n_rows, n_cols = my_params.get('axes', [1, 1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), constrained_layout=True)
    
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
            plot_data = np.where(raw_data == 0, np.nan, raw_data)
            
            target_ax = ax_secondary if (use_twin and g_idx > 0) else ax
            
            # 绘制折线
            line, = target_ax.plot(ind, plot_data, 
                                   color=data['colors'][g_idx],
                                   marker=data['markers'][g_idx], 
                                   markersize=8,
                                   linewidth=2,
                                   label=g_name, 
                                   alpha=0.8)
            handles.append(line)

            # --- OOM 标记逻辑 ---
            for j, val in enumerate(raw_data):
                if val == 0:
                    # 在折线中断处标记红色 X 或 OOM 字样
                    target_ax.text(ind[j], 0.1, xticks[j]+' OOM', 
                                   ha='center', va='bottom', 
                                   fontsize=10, color='red',
                                   fontweight='bold', transform=target_ax.get_xaxis_transform())
            
            # 针对特定轴的设置
            if use_twin and g_idx > 0:
                target_ax.set_ylabel(data.get('y2_label', g_name))
                if 'y2_lim' in data: target_ax.set_ylim(*data['y2_lim'])
                if data.get('y2_log', False): target_ax.set_yscale('log')
            else:
                ax.set_ylabel(data.get('y1_label', 'Value'))
                if 'y1_lim' in data: ax.set_ylim(*data['y1_lim'])
                if data.get('y1_log', False): ax.set_yscale('log')

        # 公共设置
        ax.set_xticks(ind)
        ax.set_xticklabels(xticks, rotation=45)
        ax.set_title(titles)
        ax.grid(True, linestyle=':', alpha=0.6)

        # 合并图例
        all_labels = [h.get_label() for h in handles]
        ax.legend(handles, all_labels, loc=data.get('legend_loc', 'upper right'), frameon=False)

    if figpath:
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.show()