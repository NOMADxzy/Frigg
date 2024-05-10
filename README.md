# Frigg: Multi-Flow Congestion Control with Mean Distribution Information in Bottleneck Link of Heterogeneous Network

## Introduce
The current heterogeneous network environment is complex
and changeable, and the interaction and competition between
flows in the bottleneck link affect the actual throughput. We
expect to explore the impact of data flow information on the
bandwidth utilization of bottleneck links. Further, we propose
a flexible congestion control strategy Frigg, utilizing shared
flow mean distribution information. It learns the strategy according to the shared information from other flows on the
bottleneck link to accurately adjust the congestion window
size for each flow without making changes in the in-network
routing, so that the overall rate matches the pipeline capacity.

![](plot_train/system.png)

## Test Result

1. draw_train.py -> plot_train
![](plot_train/train_epoch_reward.png)

2. draw_compare.py -> plot_compare
![](plot_compare/ATT-LTE-driving/flows/Throughput.png)
![](plot_compare/ATT-LTE-driving/flows/Loss.png)
![](plot_compare/ATT-LTE-driving/flows/Delay.png)
![](plot_compare/ATT-LTE-driving/flows/Utility.png)

3. draw_detail.py -> plot_detail
![](plot_detail/ATT-LTE-driving/5/delay.png)
![](plot_detail/ATT-LTE-driving/5/tput.png)
![](plot_detail/ATT-LTE-driving/5/usage.png)
![](plot_detail/ATT-LTE-driving/5/qoe.png)
![](plot_detail/ATT-LTE-driving/5/elliptic.png)

4. draw_interval_start.py -> plot_interval
![](plot_interval/one_by_one/12mbps/cubic/6/tput.png)
![](plot_interval/one_by_one/12mbps/mfg/4/tput.png)

5. pantheon results —— use pantheon origin plot results
![](results/ATT-LTE-driving/mfg/10/1/indigo_a3c_test_datalink_throughput_run1.png)