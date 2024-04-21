# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# 读取数据
delay = []
send_rate = []
delivery_rate = []
cwnd = []
seqs = []


with open('data.csv', 'r') as f:
    for line in f:
        line_splits = line.strip().split(',')  # 这里假设使用逗号作为分隔符
        delay.append(float(line_splits[0]))
        send_rate.append(float(line_splits[1]))
        delivery_rate.append(float(line_splits[2]))
        cwnd.append(float(line_splits[3]))
        seqs.append(float(line_splits[5]))

names = ["delay", "send_rate", "delivery_rate", "cwnd"]
cols = [delay, send_rate, delivery_rate, cwnd]

for i in range(0,len(names)):
    # 绘制图形
    plt.plot(seqs, cols[i], label=names[i])

    # 添加标题和标签
    plt.title(names[i])
    plt.xlabel('seq')
    plt.ylabel(names[i])
    plt.legend()
    plt.savefig("results/%s.png" % names[i])

# 显示或保存图形
# plt.show()
# 或者保存到文件
