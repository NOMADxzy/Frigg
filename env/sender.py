# -*- coding: utf-8 -*-
import os.path, copy
import time
import sys
import json
import socket
import select
from os import path
import numpy as np
import datagram_pb2
import project_root
from helpers.helpers import (
    curr_ts_ms, apply_op,
    READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, WRITE_FLAGS, ALL_FLAGS)
import indigo_pb2_grpc, grpc, indigo_pb2


def format_actions(action_list):
    """ Returns the action list, initially a list with elements "[op][val]"
    like /2.0, -3.0, +1.0, formatted as a dictionary.

    The dictionary keys are the unique indices (to retrieve the action) and
    the values are lists ['op', val], such as ['+', '2.0'].
    """
    return {idx: [action[0], float(action[1:])]
                  for idx, action in enumerate(action_list)}


class Sender(object):
    # RL exposed class/static variables
    max_steps = 1000
    state_dim = 4 # TODO 更改维度
    action_mapping = format_actions(["/2.0", "-10.0", "+0.0", "+10.0", "*2.0"])
    action_cnt = len(action_mapping)

    def __init__(self, port=0, train=False, debug=False):
        self.train = train
        self.port = port
        self.debug = debug

        # UDP socket and poller
        self.peer_addr = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        sys.stderr.write('[sender] Listening on port %s\n' %
                         self.sock.getsockname()[1])

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

        self.dummy_payload = 'x' * 1400

        if self.debug:
            self.sampling_file = open(path.join(project_root.DIR, 'env', 'sampling_time'), 'w', 0)

        # congestion control related
        self.seq_num = 0
        self.next_ack = 0
        self.cwnd = 10.0
        self.step_len_ms = 10

        channel = grpc.insecure_channel('localhost:50053')
        self.stub = indigo_pb2_grpc.acerServiceStub(channel)

        # state variables for RLCC
        self.delivered_time = 0
        self.delivered = 0
        self.sent_bytes = 0

        self.min_rtt = float('inf')
        self.delay_ewma = None
        self.send_rate_ewma = None
        self.delivery_rate_ewma = None

        self.step_start_ms = None
        self.running = True

        self.start_seq = self.seq_num
        self.arrive_cnt = 0
        self.loss_rate = 0

        if self.train:
            self.step_cnt = 0

            self.ts_first = None
            self.rtt_buf = []

        self.handshaked = False
        self.metric_data = []
        self.metric_file = os.path.join("results", "data{}.csv".format(self.port))

    def cleanup(self):
        # 回收资源
        if self.debug and self.sampling_file:
            self.sampling_file.close()
        self.sock.close()
        # 数据导出
        with open(self.metric_file, 'w') as f:
            for line in self.metric_data:
                # 将数据点转换为逗号分隔的字符串，然后写入文件
                f.write(','.join(map(str, line)) + '\n')

    def handshake(self):
        """Handshake with peer receiver. Must be called before run()."""

        while True:
            msg, addr = self.sock.recvfrom(1600)

            if msg == 'Hello from receiver' and self.peer_addr is None:
                self.peer_addr = addr
                self.sock.sendto('Hello from sender', self.peer_addr)
                sys.stderr.write('[sender] Handshake success! '
                                 'Receiver\'s address is %s:%s\n' % addr)
                break

        self.sock.setblocking(0)  # non-blocking UDP socket
        self.handshaked = True

    def set_sample_action(self, sample_action):
        """Set the policy. Must be called before run()."""

        self.sample_action = sample_action

    def update_state(self, ack):
        """ Update the state variables listed in __init__() """
        self.next_ack = max(self.next_ack, ack.seq_num + 1)
        self.arrive_cnt += 1
        curr_time_ms = curr_ts_ms()

        # Update RTT
        rtt = float(curr_time_ms - ack.send_ts)
        self.min_rtt = min(self.min_rtt, rtt)

        if self.train:
            if self.ts_first is None:
                self.ts_first = curr_time_ms
            self.rtt_buf.append(rtt)

        delay = rtt - self.min_rtt
        if self.delay_ewma is None:
            self.delay_ewma = delay
        else:
            self.delay_ewma = 0.875 * self.delay_ewma + 0.125 * delay

        # Update BBR's delivery rate
        self.delivered += ack.ack_bytes
        self.delivered_time = curr_time_ms
        delivery_rate = (0.008 * (self.delivered - ack.delivered) /
                         max(1, self.delivered_time - ack.delivered_time))

        if self.delivery_rate_ewma is None:
            self.delivery_rate_ewma = delivery_rate
        else:
            self.delivery_rate_ewma = (
                0.875 * self.delivery_rate_ewma + 0.125 * delivery_rate)

        # Update Vegas sending rate
        send_rate = 0.008 * (self.sent_bytes - ack.sent_bytes) / max(1, rtt)

        if self.send_rate_ewma is None:
            self.send_rate_ewma = send_rate
        else:
            self.send_rate_ewma = (
                0.875 * self.send_rate_ewma + 0.125 * send_rate)

    def take_action(self, action_idx):
        old_cwnd = self.cwnd
        op, val = self.action_mapping[action_idx]

        self.cwnd = apply_op(op, self.cwnd, val)
        self.cwnd = max(2.0, self.cwnd)

    def set_cwnd(self, cwnd):
        self.cwnd = max(2.0, cwnd)

    def window_is_open(self):
        return self.seq_num - self.next_ack < self.cwnd


    def send(self):
        data = datagram_pb2.Data()
        data.seq_num = self.seq_num
        data.send_ts = curr_ts_ms()
        data.sent_bytes = self.sent_bytes
        data.delivered_time = self.delivered_time
        data.delivered = self.delivered
        data.payload = self.dummy_payload

        serialized_data = data.SerializeToString()
        self.sock.sendto(serialized_data, self.peer_addr)

        self.seq_num += 1
        self.sent_bytes += len(serialized_data)

    def recv(self):
        serialized_ack, addr = self.sock.recvfrom(1600)
        k=-1

        if addr != self.peer_addr:
            return

        ack = datagram_pb2.Ack()
        ack.ParseFromString(serialized_ack)

        self.update_state(ack)

        if self.step_start_ms is None:
            self.step_start_ms = curr_ts_ms()

        # At each step end, feed the state:
        if curr_ts_ms() - self.step_start_ms > self.step_len_ms:  # step's end、
            loss_rate = 0
            if ack.seq_num - self.start_seq > 0:
                loss_rate = max(0, 1 - self.arrive_cnt / (ack.seq_num - self.start_seq))
            state = [self.delay_ewma,
                     self.delivery_rate_ewma,
                     self.send_rate_ewma,
                     self.cwnd]


            # time how long it takes to get an action from the NN
            if self.debug:
                start_sample = time.time()

            # 统计
            new_line = copy.deepcopy(state)
            new_line.append(loss_rate)
            new_line.append(ack.seq_num)
            self.metric_data.append(new_line)

            if self.train:
                input_state = self.stub.UpdateMetric(indigo_pb2.State(delay=state[0], delivery_rate=state[1], send_rate=state[2], cwnd=state[3],
                                     port=self.port))
                assert self.port == input_state.port
                state = [input_state.delay, input_state.delivery_rate, input_state.send_rate, input_state.cwnd]
                for e in input_state.nums:
                    state.append(e)
                action = self.sample_action(state[:self.state_dim])
                self.take_action(action)
            else:
                start_time = time.time()
                # 要计时的代码
                cwnd_val = self.stub.GetExplorationAction(
                    indigo_pb2.State(delay=state[0], delivery_rate=state[1], send_rate=state[2], cwnd=state[3],
                                     port=self.port)).action


                end_time = time.time()
                elapsed_time = end_time - start_time
                infer_time_file = "./infer_time.txt"
                if not os.path.exists(infer_time_file):
                    with open(infer_time_file, "w") as f:
                        f.write(str(elapsed_time))
                self.set_cwnd(cwnd_val)


            if self.debug:
                self.sampling_file.write('%.2f ms\n' % ((time.time() - start_sample) * 1000))


            self.delay_ewma = None
            self.delivery_rate_ewma = None
            self.send_rate_ewma = None
            self.start_seq = self.seq_num
            self.arrive_cnt = 0

            self.step_start_ms = curr_ts_ms()

            if self.train:
                self.step_cnt += 1
                if self.step_cnt >= Sender.max_steps:
                    self.step_cnt = 0
                    self.running = False

                    k=self.compute_performance(loss_rate)
        return k

    def run(self):
        while not self.handshaked:
            time.sleep(1)
        TIMEOUT = 1000  # ms
        sys.stderr.write("start run sender " + str(self.port) + "\n")

        self.poller.modify(self.sock, ALL_FLAGS)
        curr_flags = ALL_FLAGS
        r = -1

        while self.running:
            #print("while self.running")
            if self.window_is_open():
                if curr_flags != ALL_FLAGS:
                    self.poller.modify(self.sock, ALL_FLAGS)
                    curr_flags = ALL_FLAGS
            else:
                if curr_flags != READ_ERR_FLAGS:
                    self.poller.modify(self.sock, READ_ERR_FLAGS)
                    curr_flags = READ_ERR_FLAGS

            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                self.send()

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Error occurred to the channel')

                if flag & READ_FLAGS:
                    r = self.recv()

                if flag & WRITE_FLAGS:
                    if self.window_is_open():
                        self.send()
        return r # 返回最后一刻的奖励

    def compute_performance(self, loss_rate): # 计算奖励
        print("****************IN COMPUTE_PERFORMANCE*********************")
        duration = curr_ts_ms() - self.ts_first
        tput = 0.008 * self.delivered / duration
        perc_delay = np.percentile(self.rtt_buf, 95)
        print(tput)
        print perc_delay
        # print self.rtt_buf
        print loss_rate
        return 10*tput - perc_delay - 1000*loss_rate

        with open(path.join(project_root.DIR, 'env', 'perf'), 'a', 0) as perf:
            perf.write('%.2f %d\n' % (tput, perc_delay))
