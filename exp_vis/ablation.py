from general import parallel_bar

if __name__ == "__main__":
    params = {
        'font.family': 'Arial',
        'axes.labelsize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.titlesize': 16,
    }
    
    # 配置参数：通过 use_twin 开关控制
    my_params = {
        'axes': [1, 1],
        'bar_width': 0.3,
        'axes_params' : [
            {
                'y_val' : {
                    'GT': [1.243789474, 0, 0, 0],
                    'window': [1.6295, 17.2869, 33.8066, 3.7894],
                    'NeutronGT' : [0.1671,1.9705,1.6833,0.723],
                },
                'x_ticks' : ['arxiv','Amazon','products','reddit'],
                'title' : 'Abaltion',
                'y1_lim' : (0, 40),
                # 'y2_lim' : (1e-5, 10),
                'y1_log' : False,
                # 'y2_log' : True,
                'y1_label' : 'cost time (s)',
                # 'y2_label' : 'Capture Rate',
                'hatchs' : ['///', '...', '+++'],
                'colors' : ['#1f77b4', '#ff7f0e', '#ffff0e'],
                # 'use_twin': True,              # 是否开启双 Y 轴
                'legend_loc':'upper left'
            }
        ]
    }

    parallel_bar(params, my_params, figpath='ablation.pdf')
