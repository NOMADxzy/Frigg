import os.path

import numpy as np

import utils
import json
from draw import RunData
import config

# def combine(t_dict:{}, flows_dict):
# list_len = min([len(t_list) for t_list in t_dict.values()])

# algos = ['bbr', 'cubic', 'fillp', 'fillp_sheep', 'indigo', 'ledbat', 'pcc', 'pcc_experimental', 'scream', 'sprout', 'verus', 'vivace']
algos = ['bbr', 'cubic', 'fillp', 'fillp_sheep', 'indigo', 'pcc', 'scream', 'vivace']

traces = ['ATT-LTE-driving', 'TMobile-LTE-driving', '12mbps']


def process_delay(delay: [], x_data: []):
    new_seq = []
    new_delay = []

    metric_acc = 0
    metric_cnt = 0

    first_ts = x_data[0] * 1000
    for i in range(len(x_data) + 1):
        bin_id = 0
        if i < len(x_data):
            bin_id = utils.ms_to_bin(x_data[i] * 1000, first_ts)
        if i == len(x_data) or bin_id > utils.next_seq(new_seq):
            if metric_cnt > 0:
                new_seq.append(utils.next_seq(new_seq))
                new_delay.append(metric_acc / metric_cnt)

            if i < len(x_data):
                while utils.next_seq(new_seq) < bin_id:  # 带宽太低，需要插入一些点
                    new_seq.append(utils.next_seq(new_seq))
                    new_delay.append(new_delay[-1])
                metric_acc = delay[i]
                metric_cnt = 1
        else:
            metric_acc += delay[i]
            metric_cnt += 1

    for i in range(len(new_seq)):
        new_seq[i] = new_seq[i] * config.ms_per_bin / 1000 + int(x_data[0] * 2) / 2  # 转成秒单位

    return new_seq, new_delay


def sum_all_flow(datas: {}, data_ts: {}):
    sums = config.get_initial_val_list()
    for flow_id, data in datas.items():
        data_t = data_ts[flow_id]
        for i, val in enumerate(data):
            idx = utils.ms_to_bin(data_t[i] * 1000)
            sums[idx] += val
    return sums


loss_rates = []


def load_detail(algo, trace, flows):
    with open(os.path.join('detail_log', '{}/{}/{}_datalink_detail.json'.format(trace, flows, algo))) as f:
        detail_data = json.load(f)

    ts = detail_data['egress_t']
    tput = detail_data['egress_tput']
    loss_rate = detail_data.get('loss', 0)
    loss_rates.append(loss_rate)
    tput_res = sum_all_flow(tput, ts)
    delays = {}
    delay_ts = {}
    delay_cnt = 0
    for flow_id, data in detail_data['delay'].items():
        tmp_t, tmp_delay = process_delay(data, detail_data['delay_t'][flow_id])
        delays[flow_id] = tmp_delay
        delay_ts[flow_id] = tmp_t
        delay_cnt += 1
    delay_res = sum_all_flow(delays, delay_ts)
    for i in range(len(delay_res)):
        delay_res[i] /= delay_cnt

    qoe = config.get_initial_val_list()
    for i in range(min(len(delay_res), len(tput_res))):
        qoe[i] = utils.get_qoe(tput_res[i], delay_res[i], loss_rate)

    return tput_res, delay_res, qoe


trace = traces[2]
flow = 5

data_flow1_step10_traceATT_model80 = RunData(sender_num=5, step_len_ms=10, trace=trace,
                                             model_path='mfg')
data_flow1_step10_traceATT_model80.process_data(data_flow1_step10_traceATT_model80.sum_flow_data)
tput_lists = [data_flow1_step10_traceATT_model80.sum_flow_data.delivery_rate]
delay_lists = [data_flow1_step10_traceATT_model80.sum_flow_data.delay]
qoe_lists = [data_flow1_step10_traceATT_model80.sum_flow_data.reward]

for _, algo in enumerate(algos):
    result = load_detail(algo, trace, flow)
    tput_lists.append(result[0])
    delay_lists.append(result[1])
    qoe_lists.append(result[2])

algos.insert(0, 'mfg')

result_dir = 'plot_detail/{}/{}/'.format(trace, flow)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
utils.draw_list(tput_lists, algos, y_label='tput', save_dir=os.path.join(result_dir, 'tput.png'))
utils.draw_list(delay_lists, algos, y_label='delay', save_dir=os.path.join(result_dir, 'delay.png'))
utils.draw_list(qoe_lists, algos, y_label='qoe', save_dir=os.path.join(result_dir, 'qoe.png'))
