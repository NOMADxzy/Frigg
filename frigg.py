# -*- coding: utf-8 -*-
import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import bisect
from flow_data import FlowData
import utils,config


# 记录一次运行中所有流的表现情况
class RunData:
    def __init__(self, sender_num=5, step_len_ms=10, trace='', model_path='', start_interval=0):
        self.flow_datas = []
        self.sum_flow_data = FlowData()

        self.sender_num = sender_num
        self.step_len_ms = step_len_ms
        self.trace = trace
        self.model_path = model_path

        self.loss_rate = None
        self.tput = None
        self.delay = None
        self.useage = None

        self.useages = []

        # 本次运行的带宽情况
        self.ori_bandwidth = None
        self.ori_seq = None

        # 绘图参数
        self.ms_per_bin = config.ms_per_bin  # ms

        self.load_data()

        self.process_data(self.sum_flow_data)
        for flow_data in self.flow_datas:
            self.process_data(flow_data, interval=start_interval*flow_data.id)

    def process_data(self, flow_data, interval=0):
        x_data = flow_data.seqs
        new_seq = []
        new_delivery_rate = []
        new_delay = []
        new_send_rate = []
        new_cwnd = []
        new_loss = []
        new_reward = []

        metric_acc = [0 for _ in range(4)]
        metric_cnt = 0

        for i in range(len(flow_data.seqs)):
            flow_data.seqs[i] += 3500 + interval*1000

        first_ts = x_data[0]
        for _ in range(utils.ms_to_bin(first_ts)):  # 前期补0
            new_seq.append(utils.next_seq(new_seq))
            new_delay.append(0)
            new_delivery_rate.append(0)
            new_send_rate.append(0)
            new_cwnd.append(0)
            new_loss.append(0)
            new_reward.append(0)
        for i in range(len(x_data) + 1):
            bin_id = 0
            if i < len(x_data):
                bin_id = utils.ms_to_bin(x_data[i])
            if i == len(x_data) or bin_id > utils.next_seq(new_seq):
                new_seq.append(utils.next_seq(new_seq))
                new_delay.append(metric_acc[0] / metric_cnt)
                new_delivery_rate.append(metric_acc[1] / metric_cnt)
                new_send_rate.append(metric_acc[2] / metric_cnt)
                new_cwnd.append(metric_acc[3] / metric_cnt)
                new_loss.append((new_send_rate[-1] - new_delivery_rate[-1]) / new_send_rate[-1])
                new_reward.append(utils.get_qoe(new_delivery_rate[-1], new_delay[-1], self.loss_rate))

                if i < len(x_data):
                    while utils.next_seq(new_seq) < bin_id:  # 带宽太低，需要插入一些点
                        new_seq.append(utils.next_seq(new_seq))
                        new_delay.append(new_delay[-1])
                        new_delivery_rate.append(new_delivery_rate[-1])
                        new_send_rate.append(new_send_rate[-1])
                        new_cwnd.append(new_cwnd[-1])
                        new_loss.append(new_loss[-1])
                        new_reward.append(new_reward[-1])
                    metric_acc = [flow_data.delay[i], flow_data.delivery_rate[i], flow_data.send_rate[i],
                                  flow_data.cwnd[i]]
                    metric_cnt = 1
            else:
                metric_acc[0] += flow_data.delay[i]
                metric_acc[1] += flow_data.delivery_rate[i]
                metric_acc[2] += flow_data.send_rate[i]
                metric_acc[3] += flow_data.cwnd[i]
                metric_cnt += 1

        if self.trace == 'ATT-LTE-driving':
            while utils.next_seq(new_seq) != config.bin_cnt + 2:
                new_seq.append(utils.next_seq(new_seq))
                new_delay.append(0)
                new_delivery_rate.append(0)
                new_send_rate.append(0)
                new_cwnd.append(0)
                new_loss.append(0)
                new_reward.append(0)

        for i in range(len(new_seq)):
            new_seq[i] = new_seq[i] * self.ms_per_bin / 1000 # 转成秒单位，且初始有3s的偏移
        flow_data.seqs = new_seq
        flow_data.delay = new_delay
        flow_data.delivery_rate = new_delivery_rate
        flow_data.send_rate = new_send_rate
        flow_data.cwnd = new_cwnd
        flow_data.bytes_in_flight_rate = new_loss
        flow_data.reward = new_reward


    "'data@0&step_len_ms@10&sender_num@5&trace@ATT-LTE-driving&model_path@checkpoint-80.csv'"

    def load_data(self, load_band=False):
        # 加载所有流表现情况
        template = 'data@{}&step_len_ms@{}&sender_num@{}&trace@{}&model_path@{}{}.csv'
        for i in range(self.sender_num):
            data_file = template.format(i, self.step_len_ms, self.sender_num, self.trace, self.model_path, '')
            self.flow_datas.append(FlowData(data_file, id=i))
            if i == 0:
                self.sum_flow_data = FlowData(data_file[:-4] + '&global.csv')
                for i in range(len(self.sum_flow_data.delay)):
                    self.sum_flow_data.delay[i] /= self.sender_num

        # 加载总体metric
        result_dir = "./pantheon results/{}/{}/{}/{}/".format(self.trace, self.model_path,
                                                     self.step_len_ms, self.sender_num)
        tput, delay, loss, useage, reward = utils.read_summary(result_dir)
        self.loss_rate = loss
        self.delay = delay
        self.tput = tput
        self.useage = useage

        if load_band:
            # 加载实时带宽情况
            tail = "&fix_window_{}".format(40)
            data_file = template.format(0, self.step_len_ms, self.sender_num, self.trace, self.model_path, tail)[
                        :-4] + '&global.csv'
            bandwidth_flow_data = FlowData(data_file)
            self.ori_seq, self.ori_bandwidth = bandwidth_flow_data.seqs, bandwidth_flow_data.delivery_rate

    def get_ori_band_width(self, seq):
        left = bisect.bisect_left(self.ori_seq, seq)
        bw = self.ori_bandwidth[left]
        bw += self.ori_bandwidth[left + 1]
        bw /= 2
        return bw

    def compute_useages(self):
        for i in range(len(self.sum_flow_data.seqs)):
            seq, cur_bw = self.sum_flow_data.seqs[i], self.sum_flow_data.delivery_rate[i]
            if seq >= self.ori_seq[-1]:
                break
            bw = self.get_ori_band_width(seq)
            self.useages.append(min(1.0, cur_bw / bw))


