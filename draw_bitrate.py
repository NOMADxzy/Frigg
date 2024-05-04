# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

labels = ['15','25','30','40','50','60','75','90','100','110','120']
# Pond Books Pour

# Energy
data_120 = [1.9619527777777779,	2.987147222222222,	3.3432388888888886]
data_110 = [1.7461527777777777,	2.6726916666666667,	2.9146361111111108]
data_100 = [1.4970944444444445,	2.236686111111111,	2.3007111111111116]
data_90 = [1.196577777777778,	1.977588888888889,	1.9232777777777779]
data_75 = [ 0.8139805555555555,	1.3544194444444444,	1.367311111111111]
data_60 = [0.6439694444444445,	1.0209333333333332,	1.1378972222222223]
data_50 = [0.4858388888888889,	0.829075,	0.8676055555555556]
data_40 = [0.41054722222222223,	0.6034888888888889,	0.6997972222222222]
data_30 = [0.3018527777777778,	0.46781944444444445,	0.5338666666666667]
data_25 = [0.2671138888888889,	0.40664722222222216,	0.4480305555555556]
data_15 = [0.18322777777777777,	0.3128666666666666,	0.3186805555555556]
Energy_Consumption = [data_15,data_25,data_30,data_40,data_50,data_60,data_75,data_90,data_100,
                      data_110,data_120]

# Bitrate
bitrate_120 = [4.314140625,	4.2215234375,	 10.066123046875]
bitrate_110 = [4.11419921875,	3.89453125,	9.349716796875]
bitrate_100 = [3.934111328125,	3.8727734375,	8.57513671875]
bitrate_90 = [ 3.848154296875, 3.827783203125,8.441357421875]
bitrate_75 = [ 3.56310546875,3.819462890625,8.227998046875]
bitrate_60 = [3.495693359375,4.16609375,8.125107421875]
bitrate_50 = [3.37060546875, 4.336875,7.7743359375]
bitrate_40 = [3.244248046875,	4.226640625,7.48845703125]
bitrate_30 = [3.0187890625,	4.5643359375, 6.962255859375]
bitrate_25 = [2.942451171875,	4.866357421875,6.664267578125]
bitrate_15 = [2.953740234375,	5.0416015625,5.985224609375]
Bitrate = [bitrate_15,bitrate_25,bitrate_30,bitrate_40,bitrate_50,bitrate_60,bitrate_75,bitrate_90,bitrate_100,
                      bitrate_110,bitrate_120]
# Delay
delay_120 = [54.331,82.721,92.582]
delay_110 = [48.355,74.013,80.713]
delay_100 = [41.458,61.939,63.712]
delay_90 = [33.136,54.764,53.26]
delay_75 = [22.541,37.507,37.864]
delay_60 = [17.833,28.272,31.511]
delay_50 = [13.454,22.959,24.026]
delay_40 = [11.369,16.712,19.379]
delay_30 = [8.359,12.955,14.784]
delay_25 = [7.397, 11.261,12.407]
delay_15 = [5.074,8.664,8.825]
Delay = [delay_15,delay_25,delay_30,delay_40,delay_50,delay_60,delay_75,delay_90,delay_100,
                      delay_110,delay_120]




# plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.family'] = ['Arial']
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


# fig, ax = plt.subplots(1, 1, figsize=(4, 2.472))
# fig, ax = plt.subplots(1, 1, figsize=(8, 4.944))
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# fig, ax = plt.figure(figsize=(8, 4.944))
# 设置坐标标签字体大小
ax.set_xlabel("Frame Rate(FPS)", fontsize=20)
ax.set_ylabel(..., fontsize=20)
# # 设置图例字体大小
# ax.legend(..., fontsize=20)
# 选择如何显示刻度
# ax.xaxis.set_ticks_position(‘none’)
# ax.yaxis.set_ticks_position(‘right’)
x = np.arange(len(labels))
width = 0.24#0.15  # 每根柱子宽度

# label_font = {
#     'weight': 'bold',
#     'size': 16,
#     'family': 'simsun'
# }

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
ax.tick_params(which='major', direction='in', length=5, width=1.5, labelsize=18, bottom=False)
# labelrotation=0 标签倾斜角度
ax.tick_params(axis='x', labelsize=18, bottom=False, labelrotation=0)

ax.set_xticks(x)
# ax.set_ylim(ymin=0, ymax=40)
# 0 - 1800 ，200为一个间距
# ax.set_yticks(np.arange(0, 41, 10))
# ax.set_ylabel('Bitrate', fontdict=label_font)
ax.set_ylabel('Delay(s)') # Energy Consumption(Wh) Bitrate(Mbps) Delay(s)
ax.set_xticklabels(labels)
# ax.legend(markerscale=10,fontsize=12,prop=legend_font)
ax.legend(markerscale=16, fontsize=24)

'''
# 设置有边框和头部边框颜色为空right、top、bottom、left
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
'''

# 上下左右边框线宽
linewidth = 1.5
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_linewidth(linewidth)

    # Add some text for labels, title and custom x-axis tick labels, etc.


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# hatch 	{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
# linestyle {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
rects1 = ax.bar(x - width*6/5, [row[0] for row in Delay], width, label='Student',edgecolor='lightgoldenrodyellow',color='#000e4d',linewidth=.8,
               hatch='x')
rects2 = ax.bar(x, [row[1] for row in Delay] , width, label='Teacher',ec='#FAEBD7',color='#5c095e',
                lw=.8,hatch='o')
rects3 = ax.bar(x+ width*6/5, [row[2] for row in Delay], width, label='Student',edgecolor='g',color='#9f055e',linewidth=.8,
               hatch='/')
# rects4 = ax.bar(x + width*2/5, LB, width, label='Teacher',ec='k',color='#d5314f',
#                 lw=.8,hatch='+')
# rects5 = ax.bar(x + width*8/5, Buffer, width, label='Teacher',ec='k',color='#f66934',
#                 lw=.8,hatch='\\')
# rects6 = ax.bar(x + width*14/5, Throughput, width, label='Teacher',ec='k',color='#444e86',
#                 lw=.8,hatch='-')

l1 = plt.legend([rects1,rects2,rects3],['Pond(High SI, Low TI) ','Books(High SI, High TI)',
                                        'Pour(Low SI, Low TI)'],loc = "best", fontsize="20",ncol=1)

# autolabel(rects1)
# autolabel(rects2)
fig.tight_layout()

plt.savefig('./info.jpg',dpi=400,bbox_inches = 'tight')
#plt.savefig(r'C:\Users\admin\Pictures\Camera Roll\exam_annotate1.jpg',
#            dpi=400,bbox_inches = 'tight')

plt.show()
