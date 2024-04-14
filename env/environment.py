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
        self.flows = 3
        self.senders = []
        self.receivers = None

    def set_sample_action(self, sample_action):
        """Set the sender's policy. Must be called before calling reset()."""

        self.sample_action = sample_action

    def reset(self):
        """Must be called before running rollout()."""

        self.cleanup()

        port0 = get_open_udp_port()
        subcmds = ''
        for port in range(port0, port0+self.flows):
            # start sender as an instance of Sender class
            sys.stderr.write('Starting sender...\n')
            sender = Sender(port, train=True)
            sender.set_sample_action(self.sample_action)
            self.senders.append(sender)

            # start receiver in a subprocess
            sys.stderr.write('Starting receiver...\n')
            receiver_src = path.join(
                project_root.DIR, 'env', 'run_receiver.py')
            recv_cmd = 'python %s $MAHIMAHI_BASE %s' % (receiver_src, port)
            subcmds += recv_cmd
            if port < port0 + self.flows - 1:
                subcmds += ' & '
             # sh -c 'command1 & command2 & command3 &' 三个命令能够并行（即同时）执行
            # sys.stderr.write('$ %s\n' % cmd)


            # sender completes the handshake sent from receiver

        cmd = "%s -- sh -c '%s'" % (self.mahimahi_cmd, subcmds)
        self.receivers = Popen(cmd, preexec_fn=os.setsid, shell=True)
        for sender in self.senders:
            sender.handshake()


    def rollout(self):
        """Run sender in env, get final reward of an episode, reset sender."""

        sys.stderr.write('Obtaining an episode from environment...\n')
        rewards = 0
        for sender in self.senders:
            rewards += sender.run()
        return rewards / self.flows # 返回最后一个时刻的奖励

    def cleanup(self):
        for sender in self.senders:
            if sender:
                sender.cleanup()
        self.senders = []
        if self.receivers:
            try:
                os.killpg(os.getpgid(self.receivers.pid), signal.SIGTERM)
            except OSError as e:
                sys.stderr.write('%s\n' % e)
            finally:
                self.receiver = None
