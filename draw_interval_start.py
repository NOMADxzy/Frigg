import utils, os
from frigg import RunData

# traces = ['ATT-LTE-driving', 'TMobile-LTE-driving', '12mbps']
traces = ['12mbps']
flow = 4
algos = ['mfg']

for i in range(len(traces)):
    for _, algo in enumerate(algos):
        trace = traces[i]

        tput_lists = []

        if algo == 'mfg':
            frigg_flow_data = RunData(sender_num=flow, step_len_ms=10, trace=trace,
                                      model_path='mfg', start_interval=4)
            for flow_data in frigg_flow_data.flow_datas:
                tput_lists.append(flow_data.delivery_rate)

        result_dir = "./plot_interval/{}/{}/{}".format(trace, algo, flow)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # 趋势图
        utils.draw_list(tput_lists, algos=['flow_{}'.format(i) for i in range(flow)], y_label='Throughput',
                        save_dir=os.path.join(result_dir, 'tput.png'))