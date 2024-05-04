# -*- coding: utf-8 -*-
import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import re

####  figure 格式
plt.rcParams['font.family'] = ['Arial']
fig, ax = plt.subplots(1, 1, figsize=(8, 4.944), dpi=300)
# 设置坐标标签字体大小
ax.set_xlabel("Frame Rate(FPS)", fontsize=20)
ax.set_ylabel(..., fontsize=20)

markers = ['o', 'x', '.', ',', 'v', '<', '>', '^', '1', '2', 'p']

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
    plt.scatter(x_data, y_data, label=label)
    plt.plot(x_fine, y_smooth, label='Cubic Spline', color='red')
    plt.legend()
    plt.show()


def read_summary(data_dir):
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
        loss = loss_rate_match.group(1)
    if delay_95th_percentile_match:
        delay = delay_95th_percentile_match.group(1)

    reward = float(tput) - 0.01 * float(delay) - 0.05 * float(loss) + 20
    return float(tput), float(delay), float(loss), float(tput) / float(capacity), reward


def histogram(data_list, algo_names, xlabel, x_variables, ylabel, title, save_place=None):
    n_groups = len(data_list[0])

    # 创建一个索引数组，为每组数据分配位置
    index = np.arange(n_groups)
    bar_width = 0.2  # 柱状图的宽度

    # 绘制柱状图
    fig, ax = plt.subplots()
    for i, bar_data in enumerate(data_list):
        ax.bar(index + i * bar_width, bar_data, bar_width, label=algo_names[i])

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


def draw_list(y_lists, x_list=None, label="data", save_dir=None):
    if x_list is None:
        x_list = np.arange(len(y_lists[0]))

    plt.xlabel("time")

    for idx, y_list in enumerate(y_lists):
        plt.plot(x_list, y_list, label="15FPS", linewidth=2.5, marker=markers[idx % len(markers)],
                 markersize=8, linestyle='dashed')
        plt.ylabel(label)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc="best", fontsize="16", ncol=3)
    fig.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')

    plt.show()
