import utils, os, json
from frigg import RunData




flows = [2]  # 间隔0
traces = ['12mbps']
algos = ['mfg']


for i in range(len(traces)):
    trace = traces[i]
    for _, flow in enumerate(flows):
        for j, algo in enumerate(algos):
            tput_lists = []

            frigg_flow_data = RunData(sender_num=flow, step_len_ms=10, trace=trace,
                                      model_path='mfg', p_dir='limit_log')
            for flow_data in frigg_flow_data.flow_datas:
                tput_lists.append(flow_data.delivery_rate)
            loss_rate = frigg_flow_data.loss_rate
            delay = frigg_flow_data.delay
            useage = frigg_flow_data.useage

            result_dir = "./plot_limit/{}/{}/{}".format(trace, algo, flow)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # 趋势图
            utils.draw_list(tput_lists, algos=['Flow {}'.format(i) for i in range(flow)], y_label='吞吐量（Mbps）',
                            save_dir=os.path.join(result_dir, 'tput.png'), ncol=2)
