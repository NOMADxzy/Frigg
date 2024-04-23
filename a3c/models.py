# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers, rnn


class ActorCriticNetwork(object):
    def __init__(self, state_dim, action_cnt):
        self.states = tf.placeholder(tf.float32, [None, state_dim])

        actor_h1 = layers.relu(self.states, 8)
        actor_h2 = layers.relu(actor_h1, 8)
        self.action_scores = layers.linear(actor_h2, action_cnt)
        self.action_probs = tf.nn.softmax(self.action_scores)

        critic_h1 = layers.relu(self.states, 8)
        critic_h2 = layers.relu(critic_h1, 8)
        self.state_values = tf.reshape(layers.linear(critic_h2, 1), [-1])

        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


class ActorCriticLSTM(object):
    def __init__(self, state_dim, action_cnt):
        self.states = tf.placeholder(tf.float32, [None, state_dim]) # 分别用于存储输入状态和索引
        self.indices = tf.placeholder(tf.int32, [None])
        rnn_in = tf.expand_dims(self.states, [0])  # shape=(1, ?, state_dim)

        lstm_layers = 1
        lstm_state_dim = 256
        lstm_cell_list = []
        for i in xrange(lstm_layers):
            lstm_cell_list.append(rnn.BasicLSTMCell(lstm_state_dim))
        stacked_cell = rnn.MultiRNNCell(lstm_cell_list) # 使用 rnn.MultiRNNCell 将这些 LSTM 单元堆叠起来形成一个多层 LSTM 网络。

        # lstm_state_out: (LSTMStateTuple(c1, h1), LSTMStateTuple(c2, h2))
        # rnn_out: shape=(1, ?, lstm_state_dim), includes all h2 from the batch
        rnn_out, lstm_state_out = tf.nn.dynamic_rnn(
            stacked_cell, rnn_in, dtype=tf.float32)  # 在给定的输入序列上运行定义好的多层 LSTM 网络，获取 RNN 的输出 (rnn_out) 以及最后的状态 (lstm_state_out)。

        self.lstm_state_out = []
        for i in xrange(lstm_layers):
            self.lstm_state_out.append(
                (lstm_state_out[i].c, lstm_state_out[i].h)) # self.lstm_state_out 存储了每一层 LSTM 的状态（包括 c-state 和 h-state）
# c-state (cell state) 是 LSTM 单元的长期状态向量。它携带了网络在处理整个输入序列过程中累积的信息。c-state 的设计目的是能够跨越多个时间步长期保存信息，并可以通过门控机制（遗忘门、输入门等）来添加或移除信息，从而解决传统 RNNs 面临的梯度消失和梯度爆炸问题。
#
# h-state (hidden state)，也称为隐藏状态，是 LSTM 单元的短期状态向量。它通常用于当前时间步的输出，并作为下一个时间步的输入的一部分。h-state 与 c-state 密切相关，因为它通常是基于经过门控操作后的 cell state 生成的。
#
# LSTM 单元内部包含几种门结构来调节信息流：
#
# 遗忘门 (Forget Gate): 决定哪些信息应该从 cell state 中移除。
# 输入门 (Input Gate): 控制新信息的接收并更新 cell state。
# 输出门 (Output Gate): 根据 cell state 和当前输入决定要输出哪些信息到 hidden state。
            #
        # state output: ((c1, h1), (c2, h2))
        self.lstm_state_out = tuple(self.lstm_state_out)

        # output: shape=(?, lstm_state_dim)
        output = tf.reshape(rnn_out, [-1, lstm_state_dim])
        output = tf.gather(output, self.indices) # 从整批数据中选择特定的时间步或实例的输出

        # actor
        actor_h1 = layers.relu(output, 64) # ReLU 激活函数和全连接层构造 actor 网络
        self.action_scores = layers.linear(actor_h1, action_cnt) # 存储动作得分
        self.action_probs = tf.nn.softmax(self.action_scores) # 得到每个动作的概率

        # critic
        critic_h1 = layers.relu(output, 64) # 使用 ReLU 激活函数和全连接层构造 critic 网络
        self.state_values = tf.reshape(layers.linear(critic_h1, 1), [-1]) # 模型对当前状态的价值估计

        self.trainable_vars = tf.get_collection( # 这些变量通常是模型中的权重和偏置参数，它们会在训练过程中通过反向传播算法进行优化。
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
