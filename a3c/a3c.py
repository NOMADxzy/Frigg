# -*- coding: utf-8 -*-

import sys
import time
import project_root
import numpy as np
import tensorflow as tf
import datetime
from os import path
from models import ActorCriticLSTM
from helpers.helpers import make_sure_path_exists


def normalize_state_buf(step_state_buf):
    norm_state_buf = np.asarray(step_state_buf, dtype=np.float32)
    for i in xrange(1):
        norm_state_buf[:, i][norm_state_buf[:, i] < 1.0] = 1.0 # 小于 1.0 的值，则将这些值设置为 1.0，避免在下一步取对数时出现负无穷大或复数的情况
        norm_state_buf[:, i] = np.log(norm_state_buf[:, i]) # 替换为它们的自然对数值，使其更接近正态分布，通常可以改善机器学习模型的性能

    return norm_state_buf

# 指数加权移动平均（Exponential Weighted Moving Average，EWMA）是一种统计技术，用于分析时间序列数据。
# 它通过赋予更近期的观测值以更高的权重来计算平均值，能够反映出数据的最近趋势和模式
def ewma(data, window): # 指数加权移动平均处理
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha # 1/2
    n = data.shape[0] # 4

    pows = alpha_rev**(np.arange(n+1)) # (0,1,2,3,4)

    scale_arr = 1/pows[:-1] # 1/(0,1,2,3)
    offset = data[0]*pows[1:] # (1,2,3,4)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum() # 前缀和
    out = offset + cumsums*scale_arr[::-1]
    return out


class A3C(object):
    def __init__(self, cluster, server, task_index, env, dagger):
        # distributed tensorflow related
        self.cluster = cluster
        self.server = server
        self.task_index = task_index
        self.env = env
        self.dagger = dagger
        self.time_file = open('/tmp/sample_action_time', 'w')

        self.is_chief = (task_index == 0)
        self.worker_device = '/job:worker/task:%d' % task_index
        
        # buffers required to train
        self.action_buf = []
        self.state_buf = []

        # step counters
        self.local_step = 0

        if self.dagger:
            self.max_global_step = 2000 
            self.check_point =1500 
            self.learn_rate = 1e-3
        else:
            self.max_global_step = 30
            self.check_point = 10
            self.learn_rate = 2*1e-5

        # dimension of state and action spaces
        self.state_dim = env.state_dim
        self.action_cnt = env.action_cnt

        # must call env.set_sample_action() before env.run()
        env.set_sample_action(self.sample_action)

        # build tensorflow computation graph
        self.build_tf_graph()

        # summary related
        if self.is_chief:
            date_time = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
            self.logdir = path.join(project_root.DIR, 'a3c', 'logs', date_time)
            make_sure_path_exists(self.logdir)
            self.summary_writer = tf.summary.FileWriter(self.logdir)

        # create session
        self.session = tf.Session(self.server.target)
        self.session.run(tf.global_variables_initializer())

    def cleanup(self):
        self.env.cleanup()

    def build_tf_graph(self):
        with tf.device(tf.train.replica_device_setter(
                worker_device=self.worker_device,
                cluster=self.cluster)):
            with tf.variable_scope('global'):
                self.global_network = ActorCriticLSTM(
                    state_dim=self.state_dim, action_cnt=self.action_cnt)
                self.global_step = tf.get_variable(
                    'global_step', [], tf.int32,
                    initializer=tf.constant_initializer(0, tf.int32),
                    trainable=False)

        with tf.device(self.worker_device):
            with tf.variable_scope('local'):
                self.local_network = ActorCriticLSTM(
                    state_dim=self.state_dim, action_cnt=self.action_cnt)

            self.build_loss()

    def build_loss(self):
        print("***************************IN BUILD_LOSS**************")
        pi = self.local_network

        self.actions = tf.placeholder(tf.int32, [None])
        # cross entropy loss
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( # 交叉熵损失
            logits=pi.action_scores, labels=self.actions)

        if self.dagger:
            # regularization loss
            reg_loss = 0.0
            for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                reg_loss += tf.nn.l2_loss(x)
            reg_loss *= 0.005

            # total loss
            reduced_ce_loss = tf.reduce_mean(cross_entropy_loss)
            loss = reduced_ce_loss + reg_loss
        else:
            self.rewards = tf.placeholder(tf.float32, [None])
            self.advantages = tf.placeholder(tf.float32, [None])

            # policy loss
            policy_loss = tf.reduce_mean(cross_entropy_loss * self.advantages) # 策略损失

            # value loss
            value_loss = 0.5 * tf.reduce_mean(tf.square( # 价值损失
                self.rewards - pi.state_values))

            # add entropy to loss to encourage exploration
            log_action_probs = tf.log(pi.action_probs)
            entropy = -tf.reduce_mean(pi.action_probs * log_action_probs)

            # total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy # 总损失

        grads = tf.gradients(loss, pi.trainable_vars)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)

        # calculate gradients and apply to global network
        grads_and_vars = list(zip(grads, self.global_network.trainable_vars))
        inc_global_step = self.global_step.assign_add(1)

        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        self.train_op = tf.group(
            optimizer.apply_gradients(grads_and_vars), inc_global_step) # 将计算出的梯度应用到全局网络的参数上，同时更新全局步数

        # sync local network to global network
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip( # 将全局网络的参数复制到本地网络中
            pi.trainable_vars, self.global_network.trainable_vars)])

        # TensorBoard 看板数据
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('grad_global_norm', tf.global_norm(grads))
        tf.summary.scalar('var_global_norm', tf.global_norm(pi.trainable_vars))

        if self.dagger:
            tf.summary.scalar('reduced_ce_loss', reduced_ce_loss)
            tf.summary.scalar('reg_loss', reg_loss)
        else:
            tf.summary.scalar('policy_loss', policy_loss)
            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('entropy', entropy)
            tf.summary.scalar('reward', self.rewards[-1])

        self.summary_op = tf.summary.merge_all()

    def sample_expert_action(self, step_state_buf):
        mean_delay = np.mean(step_state_buf)

        if mean_delay < 30:
            return 0
        elif mean_delay < 50:
            return 1
        elif mean_delay < 100:
            return 2
        else:
            return 3

    def sample_action(self, step_state_buf):

        # ravel() is a faster flatten()
        flat_step_state_buf = np.asarray(step_state_buf, dtype=np.float32).ravel()

        # normalize step_state_buf and append to episode buffer
        # norm_state_buf = normalize_state_buf(step_state_buf)

        # state = EWMA of past step
        ewma_delay = ewma(flat_step_state_buf, 3)

        self.state_buf.extend([ewma_delay])
        last_index = self.indices[-1] if len(self.indices) > 0 else -1
        self.indices.append(1 + last_index)

        if self.dagger:
            expert_action = self.sample_expert_action(step_state_buf)
            self.action_buf.append(expert_action)

            # exponentially decaying sample of using expert policy
            use_expert = 0.75 ** self.local_step

            if use_expert == 0:
                return expert_action

        # run ops in local networks
        pi = self.local_network

        feed_dict = {
            pi.states: [ewma_delay],  #norm_state_buf,
            pi.indices: [0],
            #pi.lstm_state_in: pi.lstm_state_init,
        }

        if self.dagger:
            ops_to_run = [pi.action_probs]#, pi.lstm_state_out]
        else:
            ops_to_run = [pi.action_probs, pi.state_values]#, pi.lstm_state_out]

        start_time = time.time()
        ret = self.session.run(ops_to_run, feed_dict)
        elapsed_time = time.time() - start_time
        self.time_file.write('TF sample_action took: %s s.\n' % elapsed_time)

        if self.dagger:
            action_probs = ret#, lstm_state_out = ret
        else:
            action_probs, state_values = ret#, lstm_state_out = ret

        # choose an action to take and update current LSTM state
        #action = np.argmax(np.random.multinomial(1, action_probs - 1e-5))

        action = np.argmax(action_probs)
        # self.lstm_state = lstm_state_out

        if not self.dagger:
            self.action_buf.append(action)
            self.value_buf.extend(state_values)
        

        return action

    def save_model(self, check_point=None):
        if check_point is None:
            model_path = path.join(self.logdir, 'model')
        else:
            model_path = path.join(self.logdir, 'checkpoint-%d' % check_point)

        make_sure_path_exists(model_path)

        # copy global parameters to local
        self.session.run(self.sync_op)

        # save local parameters to worker-0
        saver = tf.train.Saver(self.local_network.trainable_vars)
        saver.save(self.session, model_path)
        sys.stderr.write('\nModel saved to worker-0:%s\n' % model_path)

    def rollout(self): # 用于生成回合（episode）的一个步骤序列，并计算每个步骤的优势函数（advantage values）
        print("************************IN ROLLOUT**************************")
        # reset buffers for states, actions, LSTM states, etc.
        self.state_buf = []
        self.indices = []
        self.action_buf = []
        #self.lstm_state = self.local_network.lstm_state_init

        if not self.dagger:
            self.value_buf = []

        # reset environment
        self.env.reset()

        # get an episode of rollout
        final_reward = self.env.rollout() # 执行策略并收集一系列状态、动作和奖励等数据，self.sender.run()
        print(final_reward)

        # state_buf, indices, action_buf, etc. should have been filled in
        episode_len = len(self.indices)
        # assert len(self.action_buf) == episode_len

        if not self.dagger:
            assert len(self.value_buf) == episode_len

            # 每个时间点处的奖励为下一个时间点的奖励乘以折扣因子 gamma，加上当前时间点的即时奖励
            gamma = 0.9
            if gamma == 1.0:
                self.reward_buf = np.full(episode_len, final_reward)
            else:
                self.reward_buf = np.zeros(episode_len)
                self.reward_buf[-1] = final_reward
                for i in reversed(xrange(episode_len - 1)):
                    self.reward_buf[i] = self.reward_buf[i + 1] * gamma

            # compute advantages
            self.adv_buf = self.reward_buf.astype("float32") - np.asarray(self.value_buf)

    def run(self):
        pi = self.local_network

        global_step = 0
        check_point = self.check_point
        while global_step < self.max_global_step:
            sys.stderr.write('Global step: %d\n' % global_step)

            # reset local parameters to global
            self.session.run(self.sync_op)

            # get an episode of rollout
            self.rollout()

            # train using the rollout
            summarize = self.is_chief and self.local_step % 10 == 0
 

            if summarize:
                ops_to_run = [self.train_op, self.global_step, self.summary_op]
            else:
                ops_to_run = [self.train_op, self.global_step]

            if self.dagger:
                ret = self.session.run(ops_to_run, {
                    pi.states: self.state_buf,
                    # pi.indices: self.indices,
                    self.actions: self.action_buf,
                    #pi.lstm_state_in: pi.lstm_state_init,
                })
            else:
                # 这是需要在会话中运行的操作列表。它可以包括模型的优化器更新步骤、变量评估或任何其他定义在计算图中的操作
                ret = self.session.run(ops_to_run, {
                    pi.states: self.state_buf,
                    pi.indices: self.indices, # 构建 LSTM 时选择序列中的特定时间点
                    self.actions: self.action_buf,
                    self.rewards: self.reward_buf,
                    self.advantages: self.adv_buf, # 优势函数
                    #pi.lstm_state_in: pi.lstm_state_init,
                })

            global_step = ret[1]
            self.local_step += 1

            if summarize:
                self.summary_writer.add_summary(ret[2], global_step)
                self.summary_writer.flush()

            if self.is_chief and global_step >= check_point: # 首领 保存模型
                with tf.device(self.worker_device):
                    self.save_model(check_point)
                check_point += self.check_point

        if self.is_chief: # 首领 保存模型
            with tf.device(self.worker_device):
                self.save_model()
