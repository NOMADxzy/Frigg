# -*- coding: utf-8 -*-
import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import bisect
from flow_data import FlowData
import utils


# 记录一次运行中所有流的表现情况
class RunData:
    def __init__(self, sender_num=5, step_len_ms=10, trace='', model_path=''):
        self.flow_datas = []
        self.sum_flow_data = None

        self.sender_num = sender_num
        self.step_len_ms = step_len_ms
        self.trace = trace
        self.model_path = model_path

        self.useage = []

        # 本次运行的带宽情况
        self.ori_bandwidth = None
        self.ori_seq = None

        self.load_data()

    "'data@0&step_len_ms@10&sender_num@5&trace@ATT-LTE-driving&model_path@checkpoint-80.csv'"
    def load_data(self):
        # 加载所有流表现情况
        template = 'data@{}&step_len_ms@{}&sender_num@{}&trace@{}&model_path@{}{}.csv'
        for i in range(self.sender_num):
            data_file = template.format(i, self.step_len_ms, self.sender_num, self.trace, self.model_path, '')
            self.flow_datas.append(FlowData(data_file))
            if i == 0:
                self.sum_flow_data = FlowData(data_file[:-4] + '&global.csv')

        # 加载实时带宽情况
        tail = "&fix_window_{}".format(40)
        data_file = template.format(0, self.step_len_ms, self.sender_num, self.trace, self.model_path, tail)[:-4] + '&global.csv'
        bandwidth_flow_data = FlowData(data_file)
        self.ori_seq, self.ori_bandwidth = bandwidth_flow_data.seqs, bandwidth_flow_data.delivery_rate

    def get_ori_band_width(self, seq):
        left = bisect.bisect_left(self.ori_seq, seq)
        bw = self.ori_bandwidth[left]
        bw += self.ori_bandwidth[left + 1]
        bw /= 2
        return bw

    def compute_useage(self):
        for i in range(len(self.sum_flow_data.seqs)):
            seq, cur_bw = self.sum_flow_data.seqs[i], self.sum_flow_data.delivery_rate[i]
            if seq >= self.ori_seq[-1]:
                break
            bw = self.get_ori_band_width(seq)
            self.useage.append(min(1.0, cur_bw / bw))


# band_data = RunData(sender_num=5, step_len_ms=10)
data_flow5_step10_traceATT_model80 = RunData(sender_num=5, step_len_ms=10, trace='ATT-LTE-driving',
                                             model_path='checkpoint-80')
# data_flow1_step20 = RunData('data0-step_len_ms20-sender_num1-meter_bandwidthFalse.csv',
#                             ori_band_width=band_data.delivery_rate, ori_seq=band_data.seqs)

data_flow5_step10_traceATT_model80.compute_useage()

utils.draw_list(data_list=data_flow5_step10_traceATT_model80.sum_flow_data.delivery_rate)
utils.draw_list(data_list=data_flow5_step10_traceATT_model80.useage)
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
