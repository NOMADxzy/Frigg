# -*- coding: utf-8 -*-
import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import re
import config
from matplotlib.patches import Ellipse

markers = ['o', 'x', '.', ',', 'v', '<', '>', '^', '1', '2', 'p']
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple']
hatchs = ['x', '/' '+', '\\', '-', '.', '*', '|', ]
histogram_colors = ['#000e4d', '#5c095e', '#9f055e', 'purple']

algos = ['bbr', 'cubic', 'fillp', 'fillp_sheep', 'indigo', 'pcc', 'scream', 'vivace']
algo_name_changes = {
    'bbr': 'TCP BBR',
    'cubic': 'TCP Cubic',
    'fillp': 'FillP',
    'fillp_sheep': 'FillP-Sheep',
    'indigo': 'Indigo',
    'pcc': 'PCC-Allegro',
    'scream': 'SCReAM',
    'vivace': 'PCC-Vivace',
    'sprout': 'Sprout',
    'mfg': 'Frigg',
    'pcc_experimental': 'PCC-Expr',
    'low_lstm_layer': 'Frigg w/o lstm',
    'no_field': 'Frigg w/o sharing',
    'flows': 'Flows',
    'step_len_ms': 'Frequency(ms)'
}


def ms_to_bin(ts, first_ts=config.first_second * 1000):
    return int((ts - first_ts) / config.ms_per_bin)


def next_seq(seq):
    if len(seq) == 0:
        return 0
    else:
        return seq[-1] + 1


def draw_list_simple(data_list, x_data=None, label="Data"):
    if x_data is None:
        x_data = np.arange(len(data_list))
    else:
        x_data = np.array(x_data)
    y_data = np.array(data_list)
    print(np.mean(data_list))

    # 创建一个三次样条插值函数
    spline = interp1d(x_data, y_data)

    # 在更细的网格上计算插值结果
    x_fine = np.linspace(min(x_data), max(x_data), num=300)
    y_smooth = spline(x_fine)

    # 绘制结果
    plt.scatter(x_data, y_data, label=algo_name_changes.get(label, label))
    plt.plot(x_fine, y_smooth, label='Spline', color='red')
    plt.legend()
    plt.show()


def get_qoe(tput, delay, loss, qoe_type=0):
    tput_factor = 5

    delay_penalty = 0.005
    delay = min(float(delay), 5000)

    # delay_penalty = 0.01
    # delay = min(float(delay), 500)

    loss_penalty = 500
    loss = min(float(loss), 0.05)

    prefer_factor = 3
    # reward = float(tput) - 0.01 * float(delay) - 0.05 * float(loss)
    if qoe_type == 0:
        reward = tput_factor * float(tput) - delay_penalty * float(delay) - loss_penalty * float(loss)
    elif qoe_type == 1:
        reward = prefer_factor * tput_factor * float(tput) - delay_penalty * float(delay) - loss_penalty * float(loss)
    elif qoe_type == 2:
        reward = tput_factor * float(tput) - prefer_factor * delay_penalty * float(delay) - loss_penalty * float(loss)
    elif qoe_type == 3:
        reward = tput_factor * float(tput) - delay_penalty * float(delay) - prefer_factor * loss_penalty * float(loss)
    else:
        raise ValueError

    # reward += 50
    # reward /= 5
    return reward


def sum_all_flow(datas: {}, data_ts: {}):
    sums = config.get_initial_val_list()
    for flow_id, data in datas.items():
        data_t = data_ts[flow_id]
        for i, val in enumerate(data):
            idx = ms_to_bin(data_t[i] * 1000)
            sums[idx] += val
    return sums


def read_summary(data_dir, qoe_type=0):
    # 打开并读取文件内容
    with open(os.path.join(data_dir, 'indigo_a3c_test_stats_run1.log'), 'r') as file:
        data = file.read()

    # 使用正则表达式匹配所需数据
    average_throughput_match = re.search(r'Average throughput: ([\d.]+) Mbit/s', data)
    average_capacity_match = re.search(r'Average capacity: ([\d.]+) Mbit/s', data)
    loss_rate_match = re.search(r'Loss rate: ([\d.]+)%', data)
    delay_95th_percentile_match = re.search(r'95th percentile per-packet one-way delay: ([\d.]+) ms', data)

    # 提取并打印匹配到的数据
    tput, delay, loss, capacity, reward = 0, 0, 0, 0, 0
    if average_throughput_match:
        tput = average_throughput_match.group(1)
    if average_capacity_match:
        capacity = average_capacity_match.group(1)
    if loss_rate_match:
        loss = float(loss_rate_match.group(1)) / 100
    if delay_95th_percentile_match:
        delay = delay_95th_percentile_match.group(1)

    reward = get_qoe(tput, delay, loss, qoe_type=qoe_type)
    return float(tput), float(delay), float(loss), float(tput) / float(capacity), reward


def get_usage(tput, trace):
    result_dir = "./results/{}/{}/{}/{}/".format(trace, 'mfg',
                                                 10, 5)
    tmp_tput, _, _, useage, _ = read_summary(result_dir)
    return tput / (tmp_tput / useage)


def histogram(data_list, algo_names, xlabel, x_variables, ylabel, title, save_place=None):
    n_groups = len(data_list[0])

    # 创建一个索引数组，为每组数据分配位置
    index = np.arange(n_groups)
    bar_width = 0.2  # 柱状图的宽度

    # 绘制柱状图
    fig, ax = plt.subplots()
    for i, bar_data in enumerate(data_list):
        ax.bar(index + i * bar_width, bar_data, bar_width, label=algo_name_changes.get(algo_names[i], algo_names[i]))

    # 添加标签、标题和图例
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(x_variables)
    ax.legend()
    # 显示图形
    if save_place is not None:
        plt.savefig(save_place)
    else:
        plt.show()


def draw_histogram(data_list, algo_names, xlabel, x_variables, ylabel, title, save_place=None):
    fig_width = 8
    ncol = None
    if len(x_variables) > 4:  # 是QoE对比图
        fig_width = 18
        x_variables = [algo_name_changes[x_variables[i]] for i in range(len(x_variables))]
        ncol = 4
    ####  figure 格式
    plt.rcParams['font.family'] = ['Arial']
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 4.944), dpi=300)
    # label字体
    if xlabel is not None:
        ax.set_xlabel(algo_name_changes.get(xlabel, xlabel), fontsize=20)
    ax.set_ylabel(..., fontsize=20)

    x = np.arange(len(x_variables))

    ax.tick_params(which='major', direction='in', length=5, width=1.5, labelsize=18, bottom=False)
    ax.tick_params(axis='x', labelsize=18, bottom=False, labelrotation=0)  # labelrotation=0 标签倾斜角度

    ax.set_xticks(x)
    ax.set_xticklabels(x_variables)
    ax.set_ylabel(ylabel)  # Energy Consumption(Wh) Bitrate(Mbps) Delay(s)

    if ncol is not None:
        ax.legend(markerscale=16, fontsize=24, ncol=ncol)
    else:
        ax.legend(markerscale=16, fontsize=24)
    # 上下左右边框线宽
    linewidth = 1.5
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(linewidth)

    rects_list = []
    index = np.arange(len(data_list[0]))
    bar_width = 0.2  # 柱状图的宽度
    for i, data in enumerate(data_list):
        rects = ax.bar(index + i * bar_width - bar_width * 1.5, data, bar_width, label=algo_names[i],
                       edgecolor='lightgoldenrodyellow', linewidth=.8,
                       color=histogram_colors[i % len(histogram_colors)],
                       hatch=hatchs[i % len(hatchs)])
        rects_list.append(rects)

    for i in range(len(algo_names)):
        if algo_names[i] in algo_name_changes:
            algo_names[i] = algo_name_changes[algo_names[i]]
    plt.legend(rects_list, algo_names, loc="best", fontsize="20", ncol=1)
    fig.tight_layout()

    # 显示图形
    if save_place is not None:
        plt.savefig(save_place)
    else:
        plt.show()


def draw_list(y_lists, algos, step_list=None, y_label="data", save_dir=None, ncol=3):
    ####  figure 格式
    plt.rcParams['font.family'] = ['Arial']
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.944), dpi=300)
    # label字体
    ax.set_xlabel(..., fontsize=20)
    ax.set_ylabel(..., fontsize=20)

    x_list = np.arange(0, 40, step=config.step_len)
    if step_list is not None:
        x_list = step_list

    for idx, y_list in enumerate(y_lists):
        if len(y_list) > len(x_list):
            y_list = y_list[:len(x_list)]
        plt.plot(x_list[:len(y_list)], y_list, label=algo_name_changes.get(algos[idx], algos[idx]), linewidth=2.5,
                 marker=markers[idx % len(markers)],
                 markersize=8, linestyle='dashed')

    plt.xlabel("Time(s)")
    plt.ylabel(y_label)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc="best", fontsize="16", ncol=ncol)
    fig.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')
    else:
        plt.show()


def reduce_list(lst, new_length):
    """
    输入：
    lst - 要缩短的列表
    new_length - 新的列表所需长度
    输出：
    局部均匀降低长度后的列表
    """
    num_segments = len(lst) // new_length
    reduced_list = []
    for i in range(new_length):
        segment = lst[i * num_segments:(i + 1) * num_segments]
        reduced_list.append(np.mean(segment))
    return reduced_list


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='red', edgecolor='black', alpha=0.5, **kwargs):
    if len(x) != len(y):
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=lambda_[0] * n_std * 2, height=lambda_[1] * n_std * 2,
                      angle=np.rad2deg(np.arccos(v[0, 0])),
                      edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, **kwargs)
    ax.add_patch(ellipse)


def draw_elliptic(tput_lists, delay_lists, algo_lists, save_dir=None):
    ####  figure 格式
    plt.rcParams['font.family'] = ['Arial']
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.944), dpi=300)
    # label字体
    ax.set_xlabel("RTT(ms)", fontsize=20)
    ax.set_ylabel("Throughput(Mbps)", fontsize=20)

    algo_cnts = len(algo_lists)
    for i in range(algo_cnts):
        tputs = tput_lists[i]
        delays = delay_lists[i]
        min_size = min(len(tputs), len(delays))
        algo = algo_lists[i]

        # # 绘制置信椭圆
        # confidence_ellipse(delays[:min_size], tputs[:min_size], ax, n_std=2, facecolor=colors[i % len(colors)],
        #                    edgecolor='red', alpha=0.5)
        # 用三角形标注平均值
        mean_y, mean_x = np.mean(tputs), np.mean(delays)
        ax.scatter(mean_x, mean_y, color=colors[i % len(colors)], marker=markers[i % len(markers)], s=100,
                   label=algo_name_changes.get(algo, algo))
        plt.legend(fontsize=5)

    # 设置坐标标签字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc="best", fontsize="16", ncol=3)
    fig.tight_layout()


def draw_scatter(usages, loss_rates, algo_lists, save_dir=None):
    ####  figure 格式
    plt.rcParams['font.family'] = ['Arial']
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.944), dpi=300)
    # label字体
    ax.set_xlabel("Loss Rate", fontsize=20)
    ax.set_ylabel("Bandwidth Utilization", fontsize=20)

    algo_cnts = len(algo_lists)
    for i in range(algo_cnts):
        algo = algo_lists[i]
        ax.scatter(loss_rates[i], usages[i], color=colors[i % len(colors)], marker=markers[i % len(markers)], s=100,
                   label=algo_name_changes.get(algo, algo))
        plt.legend(fontsize=5)

    # 设置坐标标签字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc="best", fontsize="16", ncol=3)
    fig.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')
    else:
        plt.show()

    if save_dir is not None:
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')
    else:
        plt.show()
