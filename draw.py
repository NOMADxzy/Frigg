# -*- coding: utf-8 -*-
import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import bisect


# 读取数据 delay,delivery_rate,send_rate,cwnd,loss_rate,seq_num,reward,infer_time,distribution
class RunData:
    def __init__(self, data_file, ori_band_width=None, ori_seq=None):
        self.delay = []
        self.delivery_rate = []
        self.send_rate = []
        self.cwnd = []
        self.loss_rate = []
        self.seqs = []
        self.reward = []
        self.infer_time = []
        self.distribution = []

        self.useage = []

        self.ori_band_width = ori_band_width
        self.ori_seq = ori_seq
        with open(os.path.join('results', data_file), 'r') as f:
            row_id = -1
            for line in f:
                row_id += 1
                if row_id == 0:
                    continue
                line_splits = line.strip().split(',')  # 这里假设使用逗号作为分隔符
                self.delay.append(float(line_splits[0]))
                self.delivery_rate.append(float(line_splits[1]))
                self.send_rate.append(float(line_splits[2]))
                self.cwnd.append(float(line_splits[3]))
                self.loss_rate.append(float(line_splits[4]))
                self.seqs.append(int(line_splits[5]))
                self.reward.append(float(line_splits[6]))
                self.infer_time.append(float(line_splits[7]))
                self.distribution.append(line_splits[8])

    def get_ori_band_width(self, seq):
        left = bisect.bisect_left(self.ori_seq, seq)
        bw = self.ori_band_width[left]
        bw += self.ori_band_width[left + 1]
        bw /= 2
        return bw

    def compute_useage(self):
        for i in range(len(self.seqs)):
            seq, cur_bw = self.seqs[i], self.delivery_rate[i]
            if seq >= self.ori_seq[-1]:
                break
            bw = self.get_ori_band_width(seq)
            self.useage.append(cur_bw / bw)


band_data = RunData('data0-step_len_ms10-sender_num1-meter_bandwidthTrue.csv')
data_flow1_step10 = RunData('data0-step_len_ms10-sender_num1-meter_bandwidthFalse.csv',
                            ori_band_width=band_data.delivery_rate, ori_seq=band_data.seqs)
data_flow1_step20 = RunData('data0-step_len_ms20-sender_num1-meter_bandwidthFalse.csv',
                            ori_band_width=band_data.delivery_rate, ori_seq=band_data.seqs)
data_flow1_step10.compute_useage()
data_flow1_step20.compute_useage()


def draw_list(data_list):
    x_data = np.arange(len(data_list))
    y_data = np.array(data_list)
    print np.mean(data_list)

    # 创建一个三次样条插值函数
    spline = interp1d(x_data, y_data)

    # 在更细的网格上计算插值结果
    x_fine = np.linspace(min(x_data), max(x_data), num=300)
    y_smooth = spline(x_fine)

    # 绘制结果
    plt.scatter(x_data, y_data, label='Data Points')
    plt.plot(x_fine, y_smooth, label='Cubic Spline', color='red')
    plt.legend()
    plt.show()


# draw_list(data_list=data_flow1_step10.useage)
# draw_list(data_list=data_flow1_step20.useage)
draw_list(data_list=band_data.delivery_rate)
draw_list(data_list=band_data.send_rate)

# names = ["delay", "send_rate", "delivery_rate", "cwnd"]
# cols = [delay, send_rate, delivery_rate, cwnd]
#
# for i in range(0, len(names)):
#     # 绘制图形
#     plt.plot(seqs, cols[i], label=names[i])
#
#     # 添加标题和标签
#     plt.title(names[i])
#     plt.xlabel('seq')
#     plt.ylabel(names[i])
#     plt.legend()
#     plt.savefig("results/%s.png" % names[i])

# 显示或保存图形
# plt.show()
# 或者保存到文件
