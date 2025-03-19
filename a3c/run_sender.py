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

INF_CWND = 10000

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
    max_cwnds = config_data.get('max_cwnds', [INF_CWND for _ in range(flows)])
    if len(max_cwnds) < flows:
        max_cwnds.extend([INF_CWND]*(flows-len(max_cwnds)))
    step_len_ms = config_data['step_len_ms']
    meter_bandwidth = config_data['meter_bandwidth']
    model_name = config_data['model_name']
    trace = config_data['trace']

    model_name_dict = {'mfg': 'checkpoint-80', 'no_field': 'model', 'low_lstm_layer': 'checkpoint-160',
                       'indigo': 'model'}
    state_dim = 7 if model_name == 'mfg' else 4

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

    wait_interval = 2
    for i, port in enumerate(range(args.port, args.port + flows)):
        # start sender as an instance of Sender class  sender_num, step_len_ms
        sender = Sender(id=i, sender_num=flows, port=port, train=False, global_state=global_state,
                        step_len_ms=step_len_ms, meter_bandwidth=meter_bandwidth, trace=trace,
                        model_name=model_name, state_dim=state_dim, wait_second=i*wait_interval, max_cwnd=max_cwnds[i])
        sender.set_sample_action(learner.sample_action)
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
