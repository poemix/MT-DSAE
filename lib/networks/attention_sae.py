# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : attention_sae.py
# @Software : PyCharm


import math
import tensorflow as tf
from lib.networks.network import Network


class AttentionSAE(Network):
    def __init__(self, n_class, n_feature, trainable=True, reuse=False):
        self.data = tf.placeholder(tf.float32, shape=[None, n_feature])
        self.label = tf.placeholder(tf.float32, shape=[None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)

        self.n_class = n_class
        self.n_feature = n_feature

        super(AttentionSAE, self).__init__(
            inputs={
                'data': self.data,
                'label': self.label,
                'keep_prob': self.keep_prob,
            },
            trainable=trainable,
            reuse=reuse
        )

    def _block(self, input, name, nb_head=8, size_head=64, trainable=True, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            output = self._attention(q=input, k=input, v=input, nb_head=nb_head, size_head=size_head,
                                     name='attention', trainable=trainable)
            output += input  # skip connection

            output = self._layer_normalize(output)  # layer normalize

            return output

    def load(self, data_path, sess, saver):
        pass

    def setup(self):
        input = self._reshape(self.data, [-1, 1, self.n_feature])  # [batch_size, 1, n_feature]
        self._multi_head_attention(q=input, k=input, v=input, nb_head=8, size_head=64, name='')


if __name__ == '__main__':
    import numpy as np

    net = AttentionStackAutoEncoder(4, 100)
    a = np.ones((4, 1, 100), dtype=np.float32)
    x = tf.placeholder(tf.float32, [4, 1, 100])

    u = net._multi_head_attention(x, x, x, nb_head=8, size_head=64, name='attention')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(u, feed_dict={x: a}).shape)
