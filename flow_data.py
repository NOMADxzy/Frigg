import os

TPUT_RWD = 1
DELAY_PENAL

class FlowData:
    def __init__(self, data_file=None):
        self.delay = []
        self.delivery_rate = []
        self.send_rate = []
        self.cwnd = []
        self.loss_rate = []
        self.seqs = []
        self.reward = []
        self.infer_time = []
        self.distribution = []

        self.reward_factor = [1, 0.1, 1000]

        if data_file is None:
            return
        self.data_file = data_file

    def load_data(self):
        with open(os.path.join('results', self.data_file), 'r') as f:
            row_id = -1
            for line in f:
                row_id += 1
                if row_id == 0:
                    continue
                line_splits = line.strip().split(',')
                self.delay.append(float(line_splits[0]))
                self.delivery_rate.append(float(line_splits[1]))
                self.send_rate.append(float(line_splits[2]))
                self.cwnd.append(float(line_splits[3]))
                self.loss_rate.append(float(line_splits[4]))
                self.seqs.append(int(line_splits[5]))
                self.reward.append(float(line_splits[6]))
                self.infer_time.append(float(line_splits[7]))
                self.distribution.append(line_splits[8])


