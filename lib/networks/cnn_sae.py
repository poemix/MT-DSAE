# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : cnn_sae.py
# @Software : PyCharm


import math
import tensorflow as tf
from lib.networks.network import Network


class CNNStackAutoEncoder(Network):
    def __init__(self, n_class, n_feature, trainable=True, reuse=False):
        self.data = tf.placeholder(tf.float32, shape=[None, n_feature])
        self.label = tf.placeholder(tf.float32, shape=[None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)

        self.n_class = n_class
        self.n_feature = n_feature

        super(CNNStackAutoEncoder, self).__init__(
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
        data = self._reshape(self.data, [-1, self.n_feature, 1])
        conv1 = self._conv1d(data, k_s=3, c_o=16, s=1, name='conv1', activation='elu')
        pool1 = self._pool1d(conv1, k_s=2, s=2, name='pool1', padding='SAME')
        conv2 = self._conv1d(pool1, k_s=3, c_o=32, s=1, name='conv2', activation='elu')
        pool2 = self._pool1d(conv2, k_s=2, s=2, name='pool2', padding='SAME')
        conv3 = self._conv1d(pool2, k_s=3, c_o=64, s=1, name='conv3', activation='elu')
        pool3 = self._pool1d(conv3, k_s=2, s=2, name='pool3', padding='SAME')
        conv4 = self._conv1d(pool3, k_s=3, c_o=64, s=1, name='conv4', activation='elu')
        print(conv4)

        shape = tf.shape(self.data)
        deconv3 = self._deconv1d(conv4, k_s=3, c_o=32, o_s=tf.stack([shape[0], math.ceil(self.n_feature/4), 32]), s=2,
                                 activation='elu', name='deconv3')
        print(deconv3)
        deconv2 = self._deconv1d(deconv3, k_s=3, c_o=16, o_s=tf.stack([shape[0], math.ceil(self.n_feature/2), 16]), s=2,
                                 activation='elu', name='deconv2')
        print(deconv2)
        deconv1 = self._deconv1d(deconv2, k_s=3, c_o=1, o_s=tf.stack([shape[0], self.n_feature, 1]), s=2,
                                 activation=False, name='deconv1')

        denc = self._reshape(deconv1, [-1, self.n_feature])
        print(deconv1)
        print(denc)

        gap = self._global_average_pool1d(conv4, keepdims=False, name='gap')
        fc1 = self._fc(gap, self.n_feature//4, activation='elu', name='fc1')
        fc2 = self._fc(fc1, self.n_feature//4, activation='elu', name='fc2')
        score = self._fc(fc2, self.n_class, activation=False, name='cls_score')
        prob = self._softmax(score, name='cls_prob')

        # add to layers
        self.layers['conv1'] = conv1
        self.layers['conv2'] = conv2
        self.layers['conv3'] = conv3

        self.layers['pool1'] = pool1
        self.layers['pool2'] = pool2
        self.layers['pool3'] = pool3

        self.layers['deconv1'] = deconv1
        self.layers['deconv2'] = deconv2
        self.layers['deconv3'] = deconv3

        self.layers['denc0'] = denc

        self.layers['cls_score'] = score
        self.layers['cls_prob'] = prob

if __name__ == '__main__':
    net = CNNStackAutoEncoder(5, 1364)


