#!/usr/bin/env python

import sys
import argparse
import project_root
import numpy as np
import tensorflow as tf
from subprocess import check_call
from os import path
from a3c import A3C
from env.environment import Environment


def prepare_traces(bandwidth):
    trace_dir = path.join(project_root.DIR, 'env')

    if type(bandwidth) == int:
        if bandwidth != 12:
            gen_trace = path.join(project_root.DIR, 'helpers',
                                  'generate_trace.py')
            cmd = ['python', gen_trace, '--output-dir', trace_dir,
                   '--bandwidth', str(bandwidth)]
            sys.stderr.write('$ %s\n' % ' '.join(cmd))
            check_call(cmd)

        uplink_trace = path.join(trace_dir, '%dmbps.trace' % bandwidth)
        downlink_trace = uplink_trace
    else:
        trace_path = path.join(trace_dir, bandwidth)
        # intentionally switch uplink and downlink traces due to sender first
        uplink_trace = trace_path + '.down'
        downlink_trace = trace_path + '.up'

    return uplink_trace, downlink_trace


def create_env(task_index):
    bandwidth = int(np.linspace(30, 60, num=4, dtype=np.int)[task_index]) # 生成数组 [30, 40, 50, 60] 并选择task_index位置的元素
    delay = 25
    queue = None

    uplink_trace, downlink_trace = prepare_traces(bandwidth)
    mm_cmd = ('mm-delay %d mm-link %s %s' %
              (delay, uplink_trace, downlink_trace))
    if queue is not None:
        mm_cmd += (' --downlink-queue=droptail '
                   '--downlink-queue-args=packets=%d' % queue)

    env = Environment(mm_cmd)
    #env.setup()
    return env


def shutdown_from_driver(driver):
    cmd = ['ssh', driver, '~/RLCC/helpers/shutdown.sh']
    check_call(cmd)

#
def run(args):
    job_name = args.job_name
    task_index = args.task_index
    sys.stderr.write('Starting job %s task %d\n' % (job_name, task_index))

    ps_hosts = args.ps_hosts.split(',') # “ps”表示参数服务器，负责存储和更新模型的参数；“worker”表示执行实际计算的工作节点。
    worker_hosts = args.worker_hosts.split(',')

    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts}) # 分布式训练：参数服务器、工作节点
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index) #

    if job_name == 'ps':
        server.join() # 如果是 "ps"（参数服务器），则执行 server.join()，该服务器进入等待状态，不断监听网络请求，直到被外部中断。
    elif job_name == 'worker':
        env = create_env(task_index)

        # 异步性 (Asynchronous): A3C 使用多个 agent 并行地在多个环境副本上执行，每个 agent 有自己的一组网络参数。
        #                       这些 agent 独立地探索空间，并周期性地将它们的梯度发送给全局网络进行异步更新。因此，
        #                       A3C 不需要经验回放缓冲区，因为异步操作引入了足够的探索。
        # 鲁棒性: 异步执行可以增加算法的稳定性，因为来自不同的agent的数据提供了更不相关的、样本效率更高的更新。
        # 资源利用: 由于并行性，A3C可以在多核CPU上有效运行，而没有必要使用GPU。
        learner = A3C( # 创建环境或者设置任务相关的配置
            cluster=cluster, # 集群配置、服务器对象、任务索引、环境和是否使用深度增强学习算法（DAGGER）为参数
            server=server,
            task_index=task_index,
            env=env,
            dagger=args.dagger)

        try:
            learner.run() # 开始训练过程
        except KeyboardInterrupt:
            pass
        finally:
            learner.cleanup()
            if args.driver is not None:
                shutdown_from_driver(args.driver)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ps-hosts', required=True, metavar='[HOSTNAME:PORT, ...]',
        help='comma-separated list of hostname:port of parameter servers')
    parser.add_argument(
        '--worker-hosts', required=True, metavar='[HOSTNAME:PORT, ...]',
        help='comma-separated list of hostname:port of workers')
    parser.add_argument('--job-name', choices=['ps', 'worker'],
                        required=True, help='ps or worker')
    parser.add_argument('--task-index', metavar='N', type=int, required=True,
                        help='index of task')
    parser.add_argument('--dagger', action='store_true',
                        help='run Dagger rather than A3C')
    parser.add_argument('--driver', help='hostname of the driver')
    args = parser.parse_args()

    # run parameter servers and workers
    run(args)


if __name__ == '__main__':
    main()
