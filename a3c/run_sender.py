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


class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars):
        with tf.variable_scope('local'):
            self.pi = ActorCriticLSTM(
                state_dim=state_dim, action_cnt=action_cnt)
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
        state_dim=Sender.state_dim,
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

    #  shared things
    senders = []
    executor = futures.ThreadPoolExecutor(max_workers=args.flows)
    global_state = GlobalState()
    model_path = path.join(project_root.DIR, 'a3c', 'logs', 'checkpoint-80')
    learner = Learner(
        state_dim=Sender.state_dim,
        action_cnt=Sender.action_cnt,
        restore_vars=model_path)

    for port in range(args.port, args.port + args.flows):
        # start sender as an instance of Sender class
        sender = Sender(port, train=False, global_state=global_state)
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
    return rewards / args.flows


if __name__ == '__main__':
    # main()
    multi_main()
