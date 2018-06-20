# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Time     : 2018/3/1 9:36
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : sae_relu.py
# @Software : PyCharm


import tensorflow as tf
from lib.networks.network import Network


class StackAutoEncoder(Network):
    def __init__(self, n_class, n_feature, trainable=True, reuse=False):
        self.data = tf.placeholder(tf.float32, shape=[None, n_feature])
        self.label = tf.placeholder(tf.float32, shape=[None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)

        self.n_class = n_class
        self.n_feature = n_feature

        super(StackAutoEncoder, self).__init__(
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
         .auto_fc(256, name='enc1', relu=True, trainable=self.trainable)
         .auto_fc(128, name='enc2', relu=True, trainable=self.trainable)
         .auto_fc(64, name='encoded', relu=True, trainable=self.trainable)
         .auto_fc(128, name='denc2', relu=True, trainable=self.trainable)
         .auto_fc(256, name='denc1', relu=True, trainable=self.trainable)
         .auto_fc(self.n_feature, name='denc0', relu=False, trainable=self.trainable))

        (self.feed('encoded')
         .auto_fc(128, name='fc1', relu=True, trainable=self.trainable)
         .auto_fc(128, name='fc2', relu=True, trainable=self.trainable)
         .auto_fc(self.n_class, name='cls_score', relu=False, trainable=self.trainable)
         .softmax(name='cls_prob'))