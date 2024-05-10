import utils, os, json
from frigg import RunData


def load_tputs(algo, trace, flows):
    """
    加载并处理特定算法、追踪和流量的数据链路详细信息

    参数:
    algo (str): 执行的算法名称。
    trace (str): 追踪的名称。
    flows (str): 流的标识符或名称。

    返回:
    list: 每个流的吞吐量列表。
    float: 资源使用率。
    float: 平均延迟。
    float: 损失率。
    """
    # 加载数据链路的详细信息
    with open(os.path.join('interval_log', '{}/{}/{}_datalink_detail.json'.format(trace, flows, algo))) as f:
        detail_data = json.load(f)

    # 提取时间戳和吞吐量数据
    ts = detail_data['egress_t']
    tput = detail_data['egress_tput']
    # 如果存在损失数据，则加载损失率，否则默认为0
    loss_rate = detail_data.get('loss', 0)
    # 计算平均使用率
    useage = utils.get_usage(detail_data.get('avg_tput', 0), trace)
    # 加载平均延迟数据，如果不存在则默认为0
    delay = detail_data.get('avg_delay', 0)
    tput_lists = []
    # 对每个流的吞吐量进行处理，累计总吞吐量
    for flow_id, flow_tput in tput.items():
        tput_res = utils.sum_all_flow({flow_id: flow_tput}, {flow_id: ts[flow_id]})
        tput_lists.append(tput_res)

    return tput_lists, useage, delay, loss_rate



# traces = ['ATT-LTE-driving', 'TMobile-LTE-driving', '12mbps']
# traces = ['12mbps']
traces = ['TMobile-LTE-driving']
# traces = ['ATT-LTE-driving']
algos = ['cubic', 'pcc', 'sprout', 'indigo']

total_results = {
    'Bandwidth Utilization': [[] for _ in range(len(algos))],
    'Delay': [[] for _ in range(len(algos))],
    'Loss': [[] for _ in range(len(algos))],
    'Utility': [[] for _ in range(len(algos))],
}

flows = [2, 4, 6, 8]  # 间隔2
# flows = [3, 5, 7, 9]  # 间隔0

for i in range(len(traces)):
    trace = traces[i]
    if flows[0] % 2 == 0:  # 偶数是依次启动的
        interval = 2
    else:
        interval = 0
    type_dir = 'one_by_one' if interval > 0 else 'together'
    for _, flow in enumerate(flows):
        for j, algo in enumerate(algos):
            tput_lists = []
            loss_rate = 0
            useage = 0

            if algo == 'mfg':
                frigg_flow_data = RunData(sender_num=flow, step_len_ms=10, trace=trace,
                                          model_path='mfg', start_interval=interval)
                for flow_data in frigg_flow_data.flow_datas:
                    tput_lists.append(flow_data.delivery_rate)
                loss_rate = frigg_flow_data.loss_rate
                delay = frigg_flow_data.delay
                useage = frigg_flow_data.useage
            else:
                tput_lists, useage, delay, loss_rate = load_tputs(algo, trace, flow)
            reward = utils.get_qoe(useage * 10, delay, loss_rate)

            total_results['Bandwidth Utilization'][j].append(useage)
            total_results['Delay'][j].append(delay)
            total_results['Loss'][j].append(loss_rate)
            total_results['Utility'][j].append(reward)

            result_dir = "./plot_interval/{}/{}/{}/{}".format(type_dir, trace, algo, flow)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # 趋势图
            utils.draw_list(tput_lists, algos=['Flow {}'.format(i) for i in range(flow)], y_label='Throughput(Mbps)',
                            save_dir=os.path.join(result_dir, 'tput.png'), ncol=2)

    result_dir = "./plot_interval/{}/{}".format(type_dir, trace)
    for y_label, data_list in total_results.items():
        utils.draw_histogram(data_list, algos, 'flows', flows, y_label,
                             '{}_{}'.format(y_label, trace),
                             os.path.join(result_dir, '{}.png'.format(y_label)))
