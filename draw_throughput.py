# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='red', edgecolor='black', alpha=0.5, **kwargs):
    if len(x) != len(y):
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=lambda_[0] * n_std * 2, height=lambda_[1] * n_std * 2,
                      angle=np.rad2deg(np.arccos(v[0, 0])),
                      edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, **kwargs)
    ax.add_patch(ellipse)


# figure格式
####  figure 格式
plt.rcParams['font.family'] = ['Arial']
fig, ax = plt.subplots(1, 1, figsize=(8, 4.944),dpi=300)

##### marker类型： 每个CC策略一个：  o x . , v < > ^ 1 2 p
# 数据1
np.random.seed(0)
data1 = np.random.normal(size=(100, 2)) * [10, 5] + [100, 50]
# 绘制置信椭圆
confidence_ellipse(data1[:, 0], data1[:, 1], ax, n_std=2, facecolor='red', edgecolor='red', alpha=0.5)
# 用三角形标注平均值
mean_x, mean_y = np.mean(data1, axis=0)
ax.scatter(mean_x, mean_y, color='red', marker='^', s=100, label = 'Cubic')

# 数据2
np.random.seed(11)
data1 = np.random.normal(size=(100, 2)) * [10, 5] + [100, 50]
# 绘制置信椭圆
confidence_ellipse(data1[:, 0], data1[:, 1], ax, n_std=2, facecolor='blue', edgecolor='red', alpha=0.5)
# 用三角形标注平均值
mean_x, mean_y = np.mean(data1, axis=0)
ax.scatter(mean_x, mean_y, color='blue', marker='o', s=100, label='BBR')
# ax.set_xlim(mean_x - 30, mean_x + 30)
# ax.set_ylim(mean_y - 30, mean_y + 30)


# 设置坐标标签字体大小
ax.set_xlabel("RTT(ms)", fontsize=20)
ax.set_ylabel("Throughput(Mbps)", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
l1 = plt.legend(loc = "best", fontsize="16",ncol=3)
fig.tight_layout()


plt.show()
