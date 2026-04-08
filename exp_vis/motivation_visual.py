import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def plot_multi_bar(plot_params, my_params, Y, labels, xlabel, ylabel, anchor=None, figpath=None):
  # print(plt.rcParams.keys())
    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42

    width = my_params['bar_width']
    colors = my_params['colors']
    hatchs = my_params['hatchs']

    fig, ax = plt.subplots()

    n = len(Y[0])
    m = len(labels)
    ind = np.arange(n)
    offset = np.arange(m) - m / 2 + 0


    for i, y in enumerate(Y):
      plt.bar(ind+(offset[i]*width),y,width,color=colors[i], hatch=hatchs[i], label=labels[i], edgecolor='black', lw=my_params['bar_line'])


    ax.set_xticks(np.arange(n), xticks, rotation=0)
    ax.tick_params(axis='x', pad=5)
    ax.set_ylim(0, 100)

    plt.legend(labels, ncol=my_params['ncol'],
                bbox_to_anchor=my_params['anchor'],
                columnspacing=my_params['columnspacing'],
                labelspacing=my_params['labelspacing'],
                handletextpad=my_params['handletextpad'],
                handleheight=my_params['handleheight'],
                handlelength=my_params['handlelength'])


    plt.xlabel(xlabel, labelpad=2)
    plt.ylabel(ylabel, labelpad=2)

    # axes = plt.gca()
    ax.spines[['right', 'top']].set_visible(True)
    ax.tick_params(bottom=True, left=True) # x,y轴的刻度线

    ax.spines['bottom'].set_linewidth(params['lines.linewidth'])
    ax.spines['left'].set_linewidth(params['lines.linewidth'])
    ax.spines['right'].set_linewidth(params['lines.linewidth'])
    ax.spines['top'].set_linewidth(params['lines.linewidth'])

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight', pad_inches=0, format='pdf')
    print(figpath, 'is plot.')
    plt.close()


def plot_multi_bar_white_hatch(plot_params, my_params, Y, labels, xlabel, ylabel, anchor=None, figpath=None):
  # print(plt.rcParams.keys())
    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42

    width = my_params['bar_width']
    colors = my_params['colors']
    hatchs = my_params['hatchs']

    fig, ax = plt.subplots()

    n = len(Y[0])
    m = len(labels)
    ind = np.arange(n)
    offset = np.arange(m) - m / 2 + 0


    h_legs, e_legs = [], []
    for i, y in enumerate(Y):
      leg1 = plt.bar(ind+(offset[i]*width),y,width,color=colors[i], hatch=hatchs[i], label=labels[i], edgecolor='white')
      leg2 = plt.bar(ind+(offset[i]*width),y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')

      h_legs.append(leg1)
      e_legs.append(leg2)


    ax.set_xticks(np.arange(n), xticks, rotation=0)
    ax.tick_params(axis='x', pad=5)
    ax.set_ylim(0, 100)

    legs = [(x,y) for x,y in zip(h_legs, e_legs)]
    plt.legend(legs, labels, ncol=my_params['ncol'],
                bbox_to_anchor=my_params['anchor'],
                columnspacing=my_params['columnspacing'],
                labelspacing=my_params['labelspacing'],
                handletextpad=my_params['handletextpad'],
                handleheight=my_params['handleheight'],
                handlelength=my_params['handlelength'])


    plt.xlabel(xlabel, labelpad=2)
    plt.ylabel(ylabel, labelpad=2)

    # axes = plt.gca()
    ax.spines[['right', 'top']].set_visible(True)
    ax.tick_params(bottom=True, left=True) # x,y轴的刻度线

    ax.spines['bottom'].set_linewidth(params['lines.linewidth'])
    ax.spines['left'].set_linewidth(params['lines.linewidth'])
    ax.spines['right'].set_linewidth(params['lines.linewidth'])
    ax.spines['top'].set_linewidth(params['lines.linewidth'])

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight', pad_inches=0, format='pdf')
    print(figpath, 'is plot.')
    plt.close()


def normalized_Y(Y):
  col_sum = Y.sum(axis=0)
  Y = Y / col_sum[np.newaxis, :]
  return Y

def plot_dual_datasets(plot_params, my_params, data1, data2, xticks, y_lim, labels, figpath=None):
    pylab.rcParams.update(plot_params)
    plt.rcParams['pdf.fonttype'] = 42

    # 创建 1行2列 的画布
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.1)
    
    datasets = [data1, data2]
    axes_left = [ax_l, ax_r]
    titles = ['pubmed', 'ogbn-arxiv'] # 可自定义标题

    for i, (ax1, data) in enumerate(zip(axes_left, datasets)):
        ind = np.arange(len(xticks[i]))
        acc_vals = data['acc']
        cap_vals = data['cap']

        # --- 左轴：Accuracy (折线图) ---
        lns1 = ax1.plot(ind, acc_vals, color=my_params['colors'][0], marker='o', 
                        label='Acc', lw=2, markersize=5)
        ax1.set_ylabel('Accuracy (%)',fontweight='bold')
        ax1.set_ylim(*y_lim[i]) # 统一 Acc 范围
        ax1.set_xticks(ind)
        ax1.set_xticklabels(xticks[i])
        ax1.set_title(titles[i], fontsize=plot_params['axes.titlesize'])
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        # --- 右轴：Capture Rate (对数柱状图) ---
        ax2 = ax1.twinx()
        bars = ax2.bar(ind, cap_vals, my_params['bar_width'], color=my_params['colors'][1], 
                       hatch=my_params['hatchs'][i], label='Capture', 
                       edgecolor='black', alpha=0.5)
        
        ax2.set_yscale('log')
        ax2.set_ylim(1e-5, 10) # 统一 Capture 范围 (10^1 为了给顶部留空)
        ax2.set_ylabel('Capture Rate (Log)')

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
    xticks = [[986, 1972, 3944, 5916], [3200, 6400, 12800, 19200, 25600]]
    y_lim = ((80, 90), (60, 70)) # 统一 Acc 范围
    
    # 数据集1
    dataset_1 = {
        'acc': [86.92, 86.84, 88.36, 88.82],
        'cap': [0.000055, 0.000248, 0.278109, 1.0]
    }
    # 数据集2
    dataset_2 = {
        'acc': [63.79, 64.55, 64.99, 66.48, 66.84],
        'cap': [0.000018, 0.000068, 0.000244, 0.749908, 1.0]
    }

    plot_dual_datasets(params, my_params, dataset_1, dataset_2, xticks, y_lim, 
                       ['Acc', 'Capture'], figpath='motivation_capture_acc_dual_comparison.pdf')
 
#   params={
#     'axes.labelsize': '11',
#     'xtick.labelsize':'11',
#     'ytick.labelsize':'11',
#     'lines.linewidth': 1,
#     'legend.fontsize': '11',
#     'figure.figsize' : '4, 2',
#     'legend.loc': 'upper center', #[]"upper right", "upper left"]
#     'legend.frameon': False,
#     'font.family': 'Arial',
#     'font.serif': 'Arial',
#   }

#   my_params={
#     'ncol': 4, # 图例列数
#     'anchor': (0.5, 1.2), # 图例位置
#     'columnspacing': 2, # 横向图例间距
#     'labelspacing': 0.5, # 纵向图例间距
#     'handletextpad': 0.1 , # 文字距离
#     'handleheight': 0.7, # 图例高度
#     'handlelength': 1.2, # 图例宽度

#     'bar_width': 0.2,
#     'bar_line': 0.5,
#     'colors': ['C3','C1','C2','C0',],
#     'hatchs': ['xx','..','**','++'],
#   }

#   Y1 = np.random.randint(0, 101, size=(4, 5))
#   Y2 = np.random.randint(0, 101, size=(4, 5))

#   Y1 = normalized_Y(Y1) * 100
#   Y2 = normalized_Y(Y2) * 100

#   labels = ['part1', 'part2', 'part3', 'part4']
#   xticks = [f'data{i}' for i in range(5)]
#   xlabel = 'Dataset'
#   ylabel = 'Norm. Execute Time (%)'

#   plot_multi_bar(params, my_params, Y1, labels, xlabel, ylabel, xticks, figpath='multi_bar.pdf')

#   plot_multi_bar_white_hatch(params, my_params, Y1, labels, xlabel, ylabel, xticks, figpath='multi_bar_white_hatch.pdf')
