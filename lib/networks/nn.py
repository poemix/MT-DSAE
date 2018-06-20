# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : nn.py
# @Software : PyCharm


import tensorflow as tf
from lib.networks.network import Network


class NeuralNetwork(Network):
    def __init__(self, n_class, n_feature, trainable=True, reuse=False):
        self.data = tf.placeholder(tf.float32, shape=[None, n_feature])
        self.label = tf.placeholder(tf.float32, shape=[None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)

        self.n_class = n_class
        self.n_feature = n_feature

        super(NeuralNetwork, self).__init__(
            inputs={
                'data': self.data,
                'label': self.label,
                'keep_prob': self.keep_prob,
            },
            trainable=trainable,
            reuse=reuse
        )

    def load(self, data_path, sess, saver):
        pass

    def setup(self):
        (self.feed('data')
         .fc(self.n_feature // 2, name='fc1', activation='relu', trainable=self.trainable)
         .fc(self.n_feature // 4, name='fc2', activation='relu', trainable=self.trainable)
         .fc(self.n_feature // 8, name='fc3', activation='relu', trainable=self.trainable)
         .fc(self.n_feature // 4, name='fc4', activation='relu', trainable=self.trainable)
         .fc(self.n_feature // 4, name='fc5', activation='relu', trainable=self.trainable)
         .fc(self.n_class, name='cls_score', activation='relu', trainable=self.trainable)
         .softmax(name='cls_prob')
         )
