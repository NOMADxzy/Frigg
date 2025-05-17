# -*- coding: utf-8 -*-
import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from env.sender import Sender
from models import ActorCriticLSTM, ActorCriticNetwork
from a3c import ewma
import socket, sys
from concurrent import futures
from env.global_state import GlobalState
import yaml
from helpers.helpers import apply_op

INF_CWND = 10000
INF_SEND_RATE = 1000
MIN_CWND = 1.0
MAX_CWND = 40.0

def format_actions(action_list):
    """ Returns the action list, initially a list with elements "[op][val]"
    like /2.0, -3.0, +1.0, formatted as a dictionary.

    The dictionary keys are the unique indices (to retrieve the action) and
    the values are lists ['op', val], such as ['+', '2.0'].
    """
    return {idx: [action[0], float(action[1:])]
            for idx, action in enumerate(action_list)}
action_mapping = format_actions(["/2.0", "-10.0", "+0.0", "+10.0", "*2.0"])
def take_action(cwnd, action_idx):
    op, val = action_mapping[action_idx]

    new_cwnd = apply_op(op, cwnd, val)
    target_cwnd = min(max(MIN_CWND, new_cwnd), MAX_CWND)
    return target_cwnd

class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars, lstm_layers=2):
        with tf.variable_scope('local'):
            self.pi = ActorCriticLSTM(
                state_dim=state_dim, action_cnt=action_cnt, lstm_layers=lstm_layers)
            # # save the current LSTM state of local network
            # self.lstm_state = self.pi.lstm_state_init

        self.session = tf.Session()

        # restore saved variables
        saver = tf.train.Saver(self.pi.trainable_vars)
        saver.restore(self.session, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables()) - set(self.pi.trainable_vars)
        self.session.run(tf.variables_initializer(uninit_vars))

    def sample_action(self, step_state_buf):
        # ravel() is a faster flatten()
        flat_step_state_buf = np.asarray(step_state_buf, dtype=np.float32).ravel()

        # state = EWMA of past step
        ewma_delay = ewma(flat_step_state_buf, 3)

        ops_to_run = [self.pi.action_probs]  # , self.pi.lstm_state_out]
        feed_dict = {
            self.pi.states: [ewma_delay],
            self.pi.indices: [0],
            # self.pi.lstm_state_in: self.lstm_state,
        }

        ret = self.session.run(ops_to_run, feed_dict)
        action_probs = ret  # , lstm_state_out = ret

        action = np.argmax(action_probs)
        # action = np.argmax(np.random.multinomial(1, action_probs[0] - 1e-5))
        # self.lstm_state = lstm_state_out
        return action

    # state: delay、delivery_rate、send_rate、cwnd，分布[3]，client_num
    def programming_action(self, state, cur_cwnd):
        # send_rate_ewma = state[2]
        # client_num = state[-1]
        mean_cwnd = state[3]

        flat_step_state_buf = np.asarray(state, dtype=np.float32).ravel()

        # state = EWMA of past step
        ewma_delay = ewma(flat_step_state_buf, 3)

        ops_to_run = [self.pi.action_probs]  # , self.pi.lstm_state_out]
        feed_dict = {
            self.pi.states: [ewma_delay],
            self.pi.indices: [0],
            # self.pi.lstm_state_in: self.lstm_state,
        }

        ret = self.session.run(ops_to_run, feed_dict)
        action_probs = ret  # , lstm_state_out = ret

        expect_action = np.argmax(action_probs)
        expect_cwnd = take_action(mean_cwnd, expect_action)

        best_action = -1
        best_delta = MAX_CWND
        for act in range(5):
            new_cwnd = take_action(cur_cwnd, act)
            delta = abs(new_cwnd - expect_cwnd)
            if delta < best_delta:
                best_delta = delta
                best_action = act

        return best_action



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    sender = Sender(args.port)

    model_path = path.join(project_root.DIR, 'a3c', 'logs', 'checkpoint-100')

    learner = Learner(
        state_dim=4,
        action_cnt=Sender.action_cnt,
        restore_vars=model_path)

    sender.set_sample_action(learner.sample_action)

    try:
        sender.handshake()
        sender.run()
    except KeyboardInterrupt:
        pass
    finally:
        sender.cleanup()


def multi_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    parser.add_argument('flows', type=int)
    args = parser.parse_args()

    #  configs
    with open('config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)
    flows = config_data['flows']
    limit_vals = config_data.get('max_cwnds', [INF_SEND_RATE for _ in range(flows)]) # no limit when there has no ‘max_cwnds’ in template.yaml
    if len(limit_vals) < flows:
        limit_vals.extend([INF_SEND_RATE]*(flows-len(limit_vals)))
    step_len_ms = config_data['step_len_ms']
    meter_bandwidth = config_data['meter_bandwidth']
    model_name = config_data['model_name']
    trace = config_data['trace']

    model_name_dict = {'mfg': 'checkpoint-80', 'no_field': 'model', 'low_lstm_layer': 'checkpoint-160',
                       'indigo': 'model', 'dynamic_programming': 'checkpoint-80'}
    state_dim = 7 if model_name == 'mfg' or model_name == 'dynamic_programming' else 4

    #  shared things
    senders = []
    executor = futures.ThreadPoolExecutor(max_workers=flows)
    global_state = GlobalState()
    model_path = path.join(project_root.DIR, 'a3c', 'logs', model_name_dict[model_name])

    learner = Learner(
        state_dim=state_dim,
        action_cnt=Sender.action_cnt,
        restore_vars=model_path,
        lstm_layers=2)

    wait_interval = 0.1 # 间隔几秒启动
    for i, port in enumerate(range(args.port, args.port + flows)):
        # start sender as an instance of Sender class  sender_num, step_len_ms
        sender = Sender(id=i, sender_num=flows, port=port, train=False, global_state=global_state,
                        step_len_ms=step_len_ms, meter_bandwidth=meter_bandwidth, trace=trace,
                        model_name=model_name, state_dim=state_dim, wait_second=i*wait_interval, max_cwnd=limit_vals[i])
        sender.set_sample_action(learner.sample_action)
        sender.set_programming_action(learner.programming_action)
        senders.append(sender)

    for sender in senders:
        sender.handshake()

    rewards = 0
    fus = []
    for sender in senders:
        sys.stderr.write("submit sender " + str(sender.port) + "\n")
        future = executor.submit(sender.run, )
        fus.append(future)
    for fu in fus:
        rewards += fu.result()
    for sender in senders:
        sender.cleanup()
    return rewards / flows


if __name__ == '__main__':
    # main()
    multi_main()
