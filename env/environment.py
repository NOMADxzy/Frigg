# -*- coding: utf-8 -*-

import os
from os import path
import sys
import signal
from subprocess import Popen
from sender import Sender
import project_root
from helpers.helpers import get_open_udp_port


class Environment(object): # 训练环境
    def __init__(self, mahimahi_cmd):
        self.mahimahi_cmd = mahimahi_cmd
        self.state_dim = Sender.state_dim
        self.action_cnt = Sender.action_cnt

        # variables below will be filled in during setup
        self.sender = None
        self.receiver = None

    def set_sample_action(self, sample_action):
        """Set the sender's policy. Must be called before calling reset()."""

        self.sample_action = sample_action

    def reset(self):
        """Must be called before running rollout()."""

        self.cleanup()

        self.port = get_open_udp_port()

        # start sender as an instance of Sender class
        sys.stderr.write('Starting sender...\n')
        self.sender = Sender(self.port, train=True)
        self.sender.set_sample_action(self.sample_action)

        # start receiver in a subprocess
        sys.stderr.write('Starting receiver...\n')
        receiver_src = path.join(
            project_root.DIR, 'env', 'run_receiver.py')
        recv_cmd = 'python %s $MAHIMAHI_BASE %s' % (receiver_src, self.port)
        cmd = "%s -- sh -c '%s'" % (self.mahimahi_cmd, recv_cmd) # sh -c 'command1 & command2 & command3 &' 三个命令能够并行（即同时）执行
        sys.stderr.write('$ %s\n' % cmd)
        self.receiver = Popen(cmd, preexec_fn=os.setsid, shell=True)

        # sender completes the handshake sent from receiver
        self.sender.handshake()

    def rollout(self):
        """Run sender in env, get final reward of an episode, reset sender."""

        sys.stderr.write('Obtaining an episode from environment...\n')
        return self.sender.run() # 返回最后一个时刻的奖励

    def cleanup(self):
        if self.sender:
            self.sender.cleanup()
            self.sender = None

        if self.receiver:
            try:
                os.killpg(os.getpgid(self.receiver.pid), signal.SIGTERM)
            except OSError as e:
                sys.stderr.write('%s\n' % e)
            finally:
                self.receiver = None
