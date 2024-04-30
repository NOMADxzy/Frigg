import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import bisect
from flow_data import FlowData
import utils

default_values = {}
all_compare_datas = {
    'trace': ['ATT-LTE-driving', 'TMobile-LTE-driving', 'Verizon-EVDO-driving'],
    'model_name': ['mfg', 'no_field', 'low_lstm_layer'],
    'step_len_ms': [10, 20, 30],
    'flows': [1, 5, 20]
}

algos_cnt = len(all_compare_datas['model_name'])
algos_names = all_compare_datas['model_name']


def compare(compare_type, compare_val_list=None):
    if compare_val_list is None:
        compare_val_list = all_compare_datas[compare_type]
    results = {
        'tput': [[] for _ in range(algos_cnt)],
        'delay': [[] for _ in range(algos_cnt)],
        'loss': [[] for _ in range(algos_cnt)]}

    for _, val in enumerate(all_compare_datas[compare_type]):
        for i, model_name in enumerate(all_compare_datas['model_name']):
            trace, step_len_ms, flows = 'ATT-LTE-driving', 10, 5
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
            tput, delay, loss = utils.read_summary(result_dir)
            results['tput'][i].append(tput)
            results['delay'][i].append(delay)
            results['loss'][i].append(loss)

    for y_label, data_list in results.items():
        utils.histogram(data_list, algos_names, compare_type, compare_val_list, y_label,
                        '{}_{}'.format(y_label, compare_type))


if __name__ == '__main__':
    compare('flows')