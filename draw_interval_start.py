import utils, os, json
from frigg import RunData


def load_tputs(algo, trace, flows):
    with open(os.path.join('interval_log', '{}/{}/{}_datalink_detail.json'.format(trace, flows, algo))) as f:
        detail_data = json.load(f)

    ts = detail_data['egress_t']
    tput = detail_data['egress_tput']
    loss_rate = detail_data.get('loss', 0)
    useage = utils.get_usage(detail_data.get('avg_tput', 0), trace)
    delay = detail_data.get('avg_delay', 0)
    tput_lists = []
    for flow_id, flow_tput in tput.items():
        tput_res = utils.sum_all_flow({flow_id: flow_tput}, {flow_id: ts[flow_id]})
        tput_lists.append(tput_res)

    return tput_lists, useage, delay, loss_rate


# traces = ['ATT-LTE-driving', 'TMobile-LTE-driving', '12mbps']
traces = ['12mbps']
flows = [2, 4, 6, 8]
algos = ['mfg', 'cubic', 'pcc', 'sprout']

total_results = {
    'Bandwidth Utilization': [[] for _ in range(len(algos))],
    'Delay': [[] for _ in range(len(algos))],
    'Loss': [[] for _ in range(len(algos))],
    'Utility': [[] for _ in range(len(algos))],
}
for i in range(len(traces)):
    trace = traces[i]
    for _, flow in enumerate(flows):
        for j, algo in enumerate(algos):

            tput_lists = []
            loss_rate = 0
            useage = 0

            if algo == 'mfg':
                frigg_flow_data = RunData(sender_num=flow, step_len_ms=10, trace=trace,
                                          model_path='mfg', start_interval=4)
                for flow_data in frigg_flow_data.flow_datas:
                    tput_lists.append(flow_data.delivery_rate)
                loss_rate = frigg_flow_data.loss_rate
                delay =frigg_flow_data.delay
                useage = frigg_flow_data.useage
            else:
                tput_lists, useage, delay, loss_rate = load_tputs(algo, trace, flow)
            reward = utils.get_qoe(useage*10, delay, loss_rate)

            total_results['Bandwidth Utilization'][j].append(useage)
            total_results['Delay'][j].append(delay)
            total_results['Loss'][j].append(loss_rate)
            total_results['Utility'][j].append(reward)

            result_dir = "./plot_interval/{}/{}/{}".format(trace, algo, flow)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # 趋势图
            utils.draw_list(tput_lists, algos=['Flow {}'.format(i) for i in range(flow)], y_label='Throughput(Mbps)',
                            save_dir=os.path.join(result_dir, 'tput.png'), ncol=2)

    result_dir = "./plot_interval/{}".format(trace)
    for y_label, data_list in total_results.items():
        utils.draw_histogram(data_list, algos, 'flows', flows, y_label,
                             '{}_{}'.format(y_label, trace),
                             os.path.join(result_dir, '{}.png'.format(y_label)))