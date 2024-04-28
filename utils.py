# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def draw_list(data_list, x_data=None):
    if x_data is None:
        x_data = np.arange(len(data_list))
    else:
        x_data = np.array(x_data)
    y_data = np.array(data_list)
    print np.mean(data_list)

    # 创建一个三次样条插值函数
    spline = interp1d(x_data, y_data)

    # 在更细的网格上计算插值结果
    x_fine = np.linspace(min(x_data), max(x_data), num=300)
    y_smooth = spline(x_fine)

    # 绘制结果
    plt.scatter(x_data, y_data, label='Data Points')
    plt.plot(x_fine, y_smooth, label='Cubic Spline', color='red')
    plt.legend()
    plt.show()
