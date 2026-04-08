import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def plot_dual_datasets(plot_params, my_params, data1, data2, xticks, y_lim, labels, figpath=None):
    pylab.rcParams.update(plot_params)
    plt.rcParams['pdf.fonttype'] = 42

    # 创建 1行2列 的画布
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.1)
    
    datasets = [data1, data2]
    axes_left = [ax_l, ax_r]
    titles = ['(a) OPT', '(b) OVA'] # 可自定义标题

    for i, (ax1, data) in enumerate(zip(axes_left, datasets)):
        ind = np.arange(len(xticks[i]))
        speedup_vals = data['Speedup']
        acc_vals = data['Acc']

        # --- 左轴：Speedup (柱状图) ---
        bars = ax1.bar(ind, speedup_vals, my_params['bar_width'], color=my_params['colors'][1], 
                       hatch=my_params['hatchs'][i], label='Speedup', edgecolor='black', alpha=0.5)
        ax1.set_ylabel('Speedup', fontweight='bold')
        ax1.set_xlabel('Window Size', fontweight='bold')
        ax1.set_ylim(0.8, 3)
        ax1.set_xticks(ind)
        ax1.set_xticklabels(xticks[i])
        ax1.set_title(titles[i], y=-0.35, fontsize=plot_params['axes.titlesize'], fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        # --- 右轴：Accuracy (折线图) ---
        ax2 = ax1.twinx()
        lns = ax2.plot(ind, acc_vals, color=my_params['colors'][0], marker='o', 
                       label='Acc', lw=2, markersize=5)
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_xlabel('Window Size', fontweight='bold')
        ax2.set_ylim(*y_lim[i])

        # 合并图例 & 上下换位 (upper -> lower)
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc='lower left', frameon=False, fontsize=plot_params['legend.fontsize'])

        # 合并图例
        # if i == 0: # 只在第一个子图添加图例
        if True:
            lines, labs = ax1.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labs + labs2, loc='upper left', frameon=False, fontsize=plot_params['legend.fontsize'])
    
    if figpath:
        plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')
    plt.show()

if __name__ == '__main__':
    params = {
        'font.family': 'Arial',
        'axes.labelsize': 18,      # 轴标签字体
        'xtick.labelsize': 15,     # x轴刻度
        'ytick.labelsize': 15,     # y轴刻度
        'legend.fontsize': 16,     # 图例字体
        'axes.titlesize': 20,      # 子图标题
        'lines.linewidth': 2,
        'font.weight': 'bold',
    }

    my_params = {
        'colors': ['#1f77b4', '#ff7f0e'], # 红蓝配色
        'hatchs': ['///', '///'],
        'bar_width': 0.5,
    }

    # 准备数据 (示例)
    xticks = [[19482, 23237, 28801, 37919, 55660], [1174, 1403, 1744, 2307, 3412]]
    y_lim = ((70, 85), (50, 60)) # 统一 Acc 范围
    
    # 数据集1
    dataset_1 = {
        'Speedup': [1, 1.00989802,1.351743519, 1.560447564,1.750519905],
        'Acc': [78.21,78.30, 78.61, 79.05, 79.96]
    }
    # 数据集2
    dataset_2 = {
        'Speedup': [1, 1.189658195, 1.473512809, 1.916690201, 2.754464286],
        'Acc': [54.33, 54.75, 54.44, 54.92, 54.63]
    }

    plot_dual_datasets(params, my_params, dataset_1, dataset_2, xticks, y_lim, 
                       ['Speedup', 'Acc'], figpath='speedup_acc_dual_comparison.pdf')
