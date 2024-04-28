import os


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

        if data_file is None:
            return

        with open(os.path.join('results', data_file), 'r') as f:
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

    def append(self, flow_data):

