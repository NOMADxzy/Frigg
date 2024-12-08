# Frigg: Multi-Flow Congestion Control with Mean Distribution Information in Bottleneck Link of Heterogeneous Network

## Introduce
Open source code for the paper "Frigg: Multi-Flow Congestion Control with Mean
Distribution Information in Bottleneck Link of
Heterogeneous Network"

Abstract—Heterogeneous networks are characterized by dense
bottleneck links, making efficient data transmission a significant
challenge. In such networks, the competition among multiple
data flows for limited bottleneck bandwidth resources results
in inefficient interactions that reduce transmission efficiency.
Moreover, the inherent dynamism of heterogeneous networks
complicates the resolution of multi-flow contention for optimal
solutions. Unlike traditional congestion control algorithms that
regulate data flow rates, we introduce a novel collaborative
congestion control framework named Frigg. This framework
utilizes Mean Distribution Information (MDI) to break down the
informational barriers between data flows and incorporates the
receiver in the control process. A distinctive advantage of Frigg is
its ability to dynamically adjust the congestion window based on
information from other flows at the bottleneck, using computed
means to circumvent the high overhead of centralized control.
Extensive real-world experiments demonstrate that Frigg signif￾icantly enhances network resource utilization in environments
with concurrent multiple flows, achieving an approximate 8%
increase in actual throughput and a 14% reduction in latency
over leading contemporary congestion control algorithms. The
variance in bandwidth distribution among multiple concurrent
flows within bottleneck is reduced by approximately 16%. Fur￾thermore, Frigg reduces memory usage, which can be deployed
at general equipment with less than 100MB, and completes
inference within 5ms, enabling real-time responses to the dynamic
challenges presented by heterogeneous networks.

[//]: # (![]&#40;plot_train/system.png&#41;)

## Usage
This is a new algorithm designed under the interface of the Pantheon framework. 
You need to clone the code of Pantheon and place this project in the third_party directory of [Pantheon](https://github.com/StanfordSNR/pantheon.git).

### Train
To train a new model, go to third_party/Frigg/a3c and run 
`python train.py --ps-hosts 127.0.0.1:9000 --worker-hosts 127.0.0.1:8000 --username YOUR USER-NAME --rlcc-dir /pantheon/third party/Frigg`. 
After training, the result will be generated in directory `third_party/Frigg/a3c/logs`

```bash
python main.py \
    --ps-hosts PS_HOSTS \
    -worker-hosts WORKER_HOSTS \
    -job-name JOB_NAME \
    -task-index TASK_INDEX \
```

In the env module, you can modify the information related to the state space, action space, and mean-field information.

### Test
Use Pantheon to test this algorithm. In addition, We have added some codes for collecting network status and congestion control status
to it, and the results will be generated in the "results" folder.
```bash
src/experiments/test.py local (--all | --schemes "<cc1> <cc2> ...")
```