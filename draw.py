# -*- coding: utf-8 -*-
import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import bisect
from flow_data import FlowData
import utils


# 读取数据 delay,delivery_rate,send_rate,cwnd,loss_rate,seq_num,reward,infer_time,distribution
class RunData:
    def __init__(self, sender_num=5, step_len_ms=10, bandwidth=False, ori_band_width=None, ori_seq=None):
        self.flow_datas = []
        self.sum_flow_data = FlowData()

        self.sender_num = sender_num
        self.step_len_ms = step_len_ms
        self.bandwidth = bandwidth

        self.useage = []

        self.ori_band_width = ori_band_width
        self.ori_seq = ori_seq

        self.load_data()

    def load_data(self):
        template = 'data{}-step_len_ms{}-sender_num{}-meter_bandwidth{}.csv'
        for i in range(self.sender_num):
            data_file = template.format(i, self.step_len_ms, self.sender_num, self.bandwidth)
            self.flow_datas.append(FlowData(data_file))
            if i == 0:
                self.sum_flow_data = FlowData(data_file[:-4] + '-global.csv')

    def get_ori_band_width(self, seq):
        left = bisect.bisect_left(self.ori_seq, seq)
        bw = self.ori_band_width[left]
        bw += self.ori_band_width[left + 1]
        bw /= 2
        return bw

    def compute_useage(self):
        for i in range(len(self.sum_flow_data.seqs)):
            seq, cur_bw = self.sum_flow_data.seqs[i], self.sum_flow_data.delivery_rate[i]
            if seq >= self.ori_seq[-1]:
                break
            bw = self.get_ori_band_width(seq)
            self.useage.append(cur_bw / bw)


# band_data = RunData(sender_num=5, step_len_ms=10)
data_flow5_step10 = RunData(sender_num=5, step_len_ms=10, bandwidth=True)
# data_flow1_step20 = RunData('data0-step_len_ms20-sender_num1-meter_bandwidthFalse.csv',
#                             ori_band_width=band_data.delivery_rate, ori_seq=band_data.seqs)
# data_flow5_step10.compute_useage()

utils.draw_list(data_list=data_flow5_step10.sum_flow_data.delivery_rate)
# utils.draw_list(data_list=band_data.send_rate)





# draw_list(data_list=data_flow1_step10.useage)
# draw_list(data_list=data_flow1_step20.useage)


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
