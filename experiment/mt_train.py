# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : mt_train.py
# @Software : PyCharm


import numpy as np
import tensorflow as tf

from lib.data_layer import data_info
from lib.networks.factory import get_network
from lib.data_layer.tianchi_data import TianchiDataLayer


learning_rate = 0.001

num_epoches = 500
batch_size = 16

mall_name = 'm_5076'
n_class = data_info.num_classes[mall_name]
n_feature = data_info.num_features[mall_name]
print(n_feature)

data_layer = TianchiDataLayer(
    data_path='../dataset/processed', m_name=mall_name,
    n_class=n_class, n_feature=n_feature, phase_train=True,
    cross_validation=True, kfold=5, kidx=0
)

net = get_network('cnn_sae', n_class=data_layer.n_class, n_feature=data_layer.n_feature)

num_batches = data_layer.db['n_train_sample'] // batch_size

X = net.data
Y = net.label
decoded = net.get_output('denc0')
y_ = net.get_output('cls_score')

total_loss = (tf.reduce_mean(tf.pow(X - decoded, 2))
              + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=Y))
              )
print(total_loss)

correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

global_step = tf.Variable(0, trainable=False)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf.reduce_mean(total_loss), global_step=global_step)
saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(num_epoches):
        epoch_losses = np.empty(0)
        for step in range((epoch * num_batches), ((epoch + 1) * num_batches)):
            blobs = data_layer.forward4train(batch_size)
            _, loss = sess.run(
                [train_op, total_loss],
                feed_dict={
                    net.data: blobs['data'],
                    net.label: blobs['label'],
                    net.keep_prob: 1.
                }
            )
            epoch_losses = np.append(epoch_losses, loss)

        train_acc = sess.run(
            accuracy,
            feed_dict={
                net.data: data_layer.db['train_data'],
                net.label: data_layer.db['train_label'],
                net.keep_prob: 1
            }
        )

        val_acc = sess.run(
            accuracy,
            feed_dict={
                net.data: data_layer.db['val_data'],
                net.label: data_layer.db['val_label'],
                net.keep_prob: 1
            }
        )
        # if epoch == 200:
        #     saver.save(sess, './model-200.ckpt')
        # if epoch == 500:
        #     saver.save(sess, './model-500.ckpt')
        print("Epoch: ", epoch,
              " Loss: ", np.mean(epoch_losses),
              " Training Accuracy: ", train_acc,
              "Validation Accuracy:", val_acc
              )