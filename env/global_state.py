import threading
from os import path
from env.sender import Sender
import numpy as np

lock = threading.Lock()
distributions = [5, 20]


def which_interval(cwnd):
    ans = 0
    for i in range(len(distributions)):
        if distributions[i] > cwnd:
            break
        ans += 1
    # ans = min(ans, len(distributions)-1)
    return ans


class GlobalState:
    def __init__(self):
        self.delay = 0.0
        self.delivery_rate = 0.0
        self.send_rate = 0.0
        self.cwnd = 0.0

        self.client_num = 0
        self.client_states = {}
        self.phi = 0.2
        self.cwnd_distributions = [0 for _ in range(len(distributions) + 1)]

    def overly(self, cur_state):
        new_state = [0, 0, 0, 0]
        avg_state = self.get_avg_state()
        new_state[0] = cur_state[0] * self.phi + avg_state[0] * (1.0 - self.phi)
        new_state[1] = cur_state[1] * self.phi + avg_state[1] * (1.0 - self.phi)
        new_state[2] = cur_state[2] * self.phi + avg_state[2] * (1.0 - self.phi)
        new_state[3] = cur_state[3] * self.phi + avg_state[3] * (1.0 - self.phi)
        return new_state

    def get_avg_state(self):
        return [self.delay / self.client_num,
                self.delivery_rate / self.client_num,
                self.send_rate / self.client_num,
                self.cwnd / self.client_num]

    def get_input_state(self, state):
        with lock:
            self.update_states(state)
            port = state[4]

            input_state = self.overly(state[:4])

            return input_state + [e / self.client_num for e in self.cwnd_distributions]

    def update_states(self, state):
        port = state[4]
        if port not in self.client_states:
            self.client_num += 1

        else:
            pre_state = self.client_states[port]
            self.delay -= pre_state[0]
            self.delivery_rate -= pre_state[1]
            self.send_rate -= pre_state[2]
            self.cwnd -= pre_state[3]

            pre_idx = which_interval(pre_state[3])
            self.cwnd_distributions[pre_idx] -= 1

        self.client_states[port] = [state[0], state[1], state[2], state[3]]
        self.delay += state[0]
        self.delivery_rate += state[1]
        self.send_rate += state[2]
        self.cwnd += state[3]
        # print [self.delay, self.delivery_rate, self.send_rate, self.cwnd]

        cur_idx = which_interval(state[3])
        self.cwnd_distributions[cur_idx] += 1

    def UpdateMetric(self, state, debug=False):

        with lock:
            self.update_states(state)
            cur_state = state[:4]
            # input_state = self.overly(cur_state)

            if debug:
                # print input_state
                print cur_state
                print self.cwnd_distributions

            # input_state.append(state[4])
            # return input_state
