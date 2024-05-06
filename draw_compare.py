# -*- coding: utf-8 -*-
import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import bisect
from flow_data import FlowData
import utils

default_values = {}
all_compare_datas = {
    'trace': ['ATT-LTE-driving', 'TMobile-LTE-driving', '12mbps'],
    'model_name': ['mfg', 'no_field', 'low_lstm_layer', 'indigo'],
    'step_len_ms': [10, 20, 30],
    'flows': [1, 5, 20, 50]
}

algos_cnt = len(all_compare_datas['model_name'])
algos_names = all_compare_datas['model_name']


def compare(compare_type, compare_val_list=None, trace='ATT-LTE-driving', save=False):
    if compare_val_list is None:
        compare_val_list = all_compare_datas[compare_type]
    results = {
        'tput': [[] for _ in range(algos_cnt)],
        'delay': [[] for _ in range(algos_cnt)],
        'loss': [[] for _ in range(algos_cnt)],
        'useage': [[] for _ in range(algos_cnt)],
        'reward': [[] for _ in range(algos_cnt)]
    }
    step_len_ms, flows = 10, 5  # 默认值

    for _, val in enumerate(all_compare_datas[compare_type]):
        for i, model_name in enumerate(all_compare_datas['model_name']):
            if compare_type == 'trace':
                trace = val
            elif compare_type == 'step_len_ms':
                step_len_ms = val
            elif compare_type == 'flows':
                flows = val
            else:
                raise ValueError('Unknown comparison type')

            result_dir = "./results/{}/{}/{}/{}/".format(trace, model_name,
                                                         step_len_ms, flows)
            tput, delay, loss, useage, reward = utils.read_summary(result_dir)
            results['tput'][i].append(tput)
            results['delay'][i].append(delay)
            results['loss'][i].append(loss)
            results['useage'][i].append(useage)
            results['reward'][i].append(reward)

    result_dir = os.path.join('plot_results', trace, compare_type)
    if save and not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for y_label, data_list in results.items():
        utils.draw_histogram(data_list, algos_names, compare_type, compare_val_list, y_label,
                        '{}_{}'.format(y_label, trace),
                        os.path.join(result_dir, '{}.png'.format(y_label)) if save else None)


if __name__ == '__main__':
    for _, trace in enumerate(all_compare_datas['trace']):
        compare('flows', trace=trace, save=True)
        compare('step_len_ms', trace=trace, save=True)
