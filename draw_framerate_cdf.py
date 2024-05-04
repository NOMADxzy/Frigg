# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import json
import statsmodels.api as sm

####  figure 格式
plt.rcParams['font.family'] = ['Arial']
fig, ax = plt.subplots(1, 1, figsize=(8, 4.944), dpi=300)
# 设置坐标标签字体大小
ax.set_xlabel("Frame Rate(FPS)", fontsize=20)
ax.set_ylabel(..., fontsize=20)
# # 设置图例字体大小
# ax.legend(..., fontsize=20)
# 选择如何显示刻度
# ax.xaxis.set_ticks_position(‘none’)
# ax.yaxis.set_ticks_position(‘right’)

# tick_params参数刻度线样式设置
# ax.tick_params(axis=‘x’, tickdir=‘in’, labelrotation=20)参数详解
# axis : 可选{‘x’, ‘y’, ‘both’} ，选择对哪个轴操作，默认是’both’
# which : 可选{‘major’, ‘minor’, ‘both’} 选择对主or副坐标轴进行操作
# direction/tickdir : 可选{‘in’, ‘out’, ‘inout’}刻度线的方向
# color : 刻度线的颜色，我一般用16进制字符串表示，eg：’#EE6363’
# width : float, 刻度线的宽度
# size/length : float, 刻度线的长度
# pad : float, 刻度线与刻度值之间的距离
# labelsize : float/str, 刻度值字体大小
# labelcolor : 刻度值颜色
# colors : 同时设置刻度线和刻度值的颜色
# bottom, top, left, right : bool, 分别表示上下左右四边，是否显示刻度线，True为显示


#### 数据读取和处理
f = open('data.json', 'r')
content = f.read()
data = json.loads(content)

# 120帧
size_120 = []  # 帧大小 (bytes)
for item in data['120']:
    for info in data['120'][item]['frames']:
        size_120.append(info['size'] / 1024 / 1024)  # 单位MB

# 110帧
size_110 = []  # 帧大小 (bytes)
for item in data['110']:
    for info in data['110'][item]['frames']:
        size_110.append(info['size'] / 1024 / 1024)

# 100帧
size_100 = []  # 帧大小 (bytes)
for item in data['100']:
    for info in data['100'][item]['frames']:
        size_100.append(info['size'] / 1024 / 1024)

# 90帧
size_90 = []  # 帧大小 (bytes)
for item in data['90']:
    for info in data['90'][item]['frames']:
        size_90.append(info['size'] / 1024 / 1024)

# 75帧
size_75 = []  # 帧大小 (bytes)
for item in data['75']:
    for info in data['75'][item]['frames']:
        size_75.append(info['size'] / 1024 / 1024)

# 60帧
size_60 = []  # 帧大小 (bytes)
for item in data['60']:
    for info in data['60'][item]['frames']:
        size_60.append(info['size'] / 1024 / 1024)

# 50帧
size_50 = []  # 帧大小 (bytes)
for item in data['50']:
    for info in data['50'][item]['frames']:
        size_50.append(info['size'] / 1024 / 1024)

# 40帧
size_40 = []  # 帧大小 (bytes)
for item in data['40']:
    for info in data['40'][item]['frames']:
        size_40.append(info['size'] / 1024 / 1024)

# 30帧
size_30 = []  # 帧大小 (bytes)
for item in data['30']:
    for info in data['30'][item]['frames']:
        size_30.append(info['size'] / 1024 / 1024)

# 25帧
size_25 = []  # 帧大小 (bytes)
for item in data['25']:
    for info in data['25'][item]['frames']:
        size_25.append(info['size'] / 1024 / 1024)

# 15帧
size_15 = []  # 帧大小 (bytes)
for item in data['15']:
    for info in data['15'][item]['frames']:
        size_15.append(info['size'] / 1024 / 1024)

# size_120 = [0.32 if x >0.32 else x for x in size_120]
# size_110 = [0.32 if x >0.32 else x for x in size_110]
# size_100 = [0.32 if x >0.32 else x for x in size_100]
# size_90 = [0.32 if x >0.32 else x for x in size_90]
# size_75 = [0.32 if x >0.32 else x for x in size_75]
# size_60 = [0.32 if x >0.32 else x for x in size_60]
# size_50 = [0.32 if x >0.32 else x for x in size_50]
# size_40 = [0.32 if x >0.32 else x for x in size_40]
# size_30 = [0.32 if x >0.32 else x for x in size_30]
# size_25 = [0.32 if x >0.32 else x for x in size_25]
# size_15 = [0.32 if x >0.32 else x for x in size_15]


# =============绘制cdf图===============
ecdf_120 = sm.distributions.ECDF(size_120)
# 等差数列，用于绘制X轴数据
x_120 = np.linspace(min(size_120), max(size_120))
# x轴数据上值对应的累计密度概率
y_120 = ecdf_120(x_120)

ecdf_110 = sm.distributions.ECDF(size_110)
x_110 = np.linspace(min(size_110), max(size_110))
y_110 = ecdf_110(x_110)

ecdf_100 = sm.distributions.ECDF(size_100)
x_100 = np.linspace(min(size_100), max(size_100))
y_100 = ecdf_100(x_100)

ecdf_90 = sm.distributions.ECDF(size_90)
x_90 = np.linspace(min(size_90), max(size_90))
y_90 = ecdf_90(x_90)

ecdf_75 = sm.distributions.ECDF(size_75)
x_75 = np.linspace(min(size_75), max(size_75))
y_75 = ecdf_75(x_75)

ecdf_60 = sm.distributions.ECDF(size_60)
x_60 = np.linspace(min(size_60), max(size_60))
y_60 = ecdf_60(x_60)

ecdf_50 = sm.distributions.ECDF(size_50)
x_50 = np.linspace(min(size_50), max(size_50))
y_50 = ecdf_50(x_50)

ecdf_40 = sm.distributions.ECDF(size_40)
x_40 = np.linspace(min(size_40), max(size_40))
y_40 = ecdf_40(x_40)

ecdf_30 = sm.distributions.ECDF(size_30)
x_30 = np.linspace(min(size_30), max(size_30))
y_30 = ecdf_30(x_30)

ecdf_25 = sm.distributions.ECDF(size_25)
x_25 = np.linspace(min(size_25), max(size_25))
y_25 = ecdf_25(x_25)

ecdf_15 = sm.distributions.ECDF(size_15)
x_15 = np.linspace(min(size_15), max(size_15))
y_15 = ecdf_15(x_15)

#### 线型
linestyle_str = [
    ('solid', 'solid'),  # Same as (0, ()) or '-'；solid’， (0, ()) ， '-'三种都代表实线。
    ('dotted', 'dotted'),  # Same as (0, (1, 1)) or '.'
    ('dashed', 'dashed'),  # Same as '--'
    ('dashdot', 'dashdot')]  # Same as '-.'

# linestyle_tuple = [
# ('loosely dotted', (0, (1, 10))),
# ('dotted', (0, (1, 1))),
# ('densely dotted', (0, (1, 2))),
# ('loosely dashed', (0, (5, 10))),
# ('dashed', (0, (5, 5))),
# ('densely dashed', (0, (5, 1))),
# ('loosely dashdotted', (0, (3, 10, 1, 10))),
# ('dashdotted', (0, (3, 5, 1, 5))),
# ('densely dashdotted', (0, (3, 1, 1, 1))),
# ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
# ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
# ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

# plt.plot(x,y,label="pdf"

line_w = 2.5
marker_s = 8
plt.plot(x_120, y_120, label="120FPS", linewidth=line_w, marker='o', markersize=marker_s, linestyle='dotted')
plt.plot(x_110, y_110, label="110FPS", linewidth=line_w, marker='x', markersize=marker_s, linestyle='solid')
plt.plot(x_100, y_100, label="100FPS", linewidth=line_w, marker='.', markersize=marker_s, linestyle='dashed')
plt.plot(x_90, y_90, label="90FPS", linewidth=line_w, marker=',', markersize=marker_s, linestyle='dashdot')
plt.plot(x_75, y_75, label="75FPS", linewidth=line_w, marker='v', markersize=marker_s, linestyle='dotted')
plt.plot(x_60, y_60, label="60FPS", linewidth=line_w, marker='<', markersize=marker_s, linestyle='dashdot')
plt.plot(x_50, y_50, label="50FPS", linewidth=line_w, marker='>', markersize=marker_s, linestyle='dashed')
plt.plot(x_40, y_40, label="40FPS", linewidth=line_w, marker='^', markersize=marker_s, linestyle='dashdot')
plt.plot(x_30, y_30, label="30FPS", linewidth=line_w, marker='1', markersize=marker_s, linestyle='dotted')
plt.plot(x_25, y_25, label="25FPS", linewidth=line_w, marker='2', markersize=marker_s, linestyle='solid')
plt.plot(x_15, y_15, label="15FPS", linewidth=line_w, marker='p', markersize=marker_s, linestyle='dashed')

plt.xlabel("Frame Size(MB)")
plt.ylabel("Proportion")

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# plt.title("CDF for continuous distribution")
# plt.legend()

l1 = plt.legend(loc="best", fontsize="16", ncol=3)

# autolabel(rects1)
# autolabel(rects2)
fig.tight_layout()

plt.savefig('./CDF.jpg', dpi=400, bbox_inches='tight')
# plt.savefig(r'C:\Users\admin\Pictures\Camera Roll\exam_annotate1.jpg',
#            dpi=400,bbox_inches = 'tight')

plt.show()
