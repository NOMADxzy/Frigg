#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import signal
import argparse
import project_root
from os import path
from subprocess import Popen, call
from helpers.helpers import get_open_udp_port

# 参数服务器 (ps):
# 职责: 参数服务器负责存储和更新模型参数。在典型的分布式训练设置中，每个参数服务器维护一部分全局模型参数的状态。
#
# 操作: 当 worker 节点完成一部分数据的处理，并计算出梯度后，它们会将这些梯度发送到对应的参数服务器。参数服务器使用这些梯度来更新它存储的模型参数。
#
# 目的: 通过使用参数服务器，模型训练可以跨多个 worker 节点进行扩展，而模型参数保持一致和同步。
#
# Worker:
# 职责: 工作节点（worker）主要负责执行计算任务，包括前向传播、反向传播和梯度计算。
#
# 操作: 在每次迭代中，worker 从参数服务器获取当前的模型参数，计算它们自己分配到的数据批次上的梯度，然后将这些梯度发送回参数服务器以更新模型。
#
# 目的: 每个 worker 可以独立地处理输入数据的不同子集，从而实现数据并行性，并加速整体训练过程。
def run(args):
    # run worker.py on ps and worker hosts
    for job_name in ['ps', 'worker']:
        host_list = args[job_name + '_list']
        procs = args[job_name + '_procs']

        for i in xrange(len(host_list)): # 不同的host有不同的带宽，[30,40,50,60]
            ssh_cmd = ['ssh', host_list[i]]

            cmd = ['python', args['worker_src'],
                   '--ps-hosts', args['ps_hosts'],
                   '--worker-hosts', args['worker_hosts'],
                   '--job-name', job_name,
                   '--task-index', str(i)]
            if args['dagger']:
                cmd.append('--dagger')
            if args['driver'] is not None:
                cmd += ['--driver', args['driver']]

            cmd = ssh_cmd + cmd

            sys.stderr.write('$ %s\n' % ' '.join(cmd))
            procs.append(Popen(cmd, preexec_fn=os.setsid))

    # ps will block forever
    for ps_proc in args['ps_procs']:
        ps_proc.communicate()


def cleanup(args):
    all_procs = args['ps_procs'] + args['worker_procs']
    for proc in all_procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError as e:
            sys.stderr.write('%s\n' % e)

    host_set = set(args['ps_list'] + args['worker_list'])
    pkill_script = path.join(args['rlcc_dir'], 'helpers', 'pkill.py')

    for host in host_set:
        kill_cmd = ['ssh', host, 'python', pkill_script, args['rlcc_dir']]
        sys.stderr.write('$ %s\n' % ' '.join(kill_cmd))
        call(kill_cmd)

    sys.stderr.write('\nAll cleaned up.\n')


def construct_args(prog_args):
    # construct a dictionary of arguments
    args = {}

    # file paths
    args['rlcc_dir'] = prog_args.rlcc_dir
    args['worker_src'] = path.join(args['rlcc_dir'], 'a3c', 'worker.py')

    # hostnames and processes
    args['ps_hosts'] = prog_args.ps_hosts
    args['worker_hosts'] = prog_args.worker_hosts

    args['ps_list'] = prog_args.ps_hosts.split(',')
    args['worker_list'] = prog_args.worker_hosts.split(',')
    args['username'] = prog_args.username

    for i, host in enumerate(args['ps_list']):
        args['ps_list'][i] = args['username'] + '@' + host.split(':')[0]

    for i, host in enumerate(args['worker_list']):
        args['worker_list'][i] = args['username'] + '@' + host.split(':')[0]

    args['ps_procs'] = []
    args['worker_procs'] = []
    args['dagger'] = prog_args.dagger
    args['driver'] = prog_args.driver

    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ps-hosts', required=True, metavar='[HOSTNAME:PORT, ...]',
        help='comma-separated list of hostname:port of parameter servers')
    parser.add_argument(
        '--worker-hosts', required=True, metavar='[HOSTNAME:PORT, ...]',
        help='comma-separated list of hostname:port of workers')
    parser.add_argument(
        '--username', default='ubuntu',
        help='username used in ssh connection (default: ubuntu)')
    parser.add_argument(
        '--rlcc-dir', metavar='DIR', default='/home/ubuntu/RLCC',
        help='absolute path to RLCC/ (default: /home/ubuntu/RLCC)')
    parser.add_argument('--dagger', action='store_true',
        help='run Dagger rather than A3C')
    parser.add_argument('--driver', help='hostname of the driver')
    prog_args = parser.parse_args()
    args = construct_args(prog_args)

    # run worker.py on ps and worker hosts
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(args)


if __name__ == '__main__':
    main()
