# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : dsae_train.py
# @Software : PyCharm


import numpy as np
import tensorflow as tf
from lib.datasets.tdata import TData
from lib.datasets.data_info import DataInfo as di
from lib.networks.factory import get_network
from experiment.hyperparams import HyperParams as hp


def train_net(mall_name, net_name='dsae', kidx=0):
    """
    train network
    :param mall_name: mall name
    :param net_name: network name
    :param kidx: k'th k-flod cross validation
    :return: 
    """
    n_class = di.num_classes[mall_name]
    n_feature = di.num_features[mall_name]
    class_map = di.classes[mall_name]
    file_path = hp.file_path.format(mall_name)
    print(n_class, n_feature)

    # data
    tdata = TData(file_path=file_path, n_feature=n_feature, n_class=n_class, class_map=class_map,
                  kidx=kidx, kfold=hp.kfold)
    tx, ty, vx, vy, tnum_batch, vnum_batch = tdata.get_batch_data()
    is_training = tf.placeholder(tf.bool)

    data = tf.cond(is_training, lambda: tx, lambda: vx)
    label = tf.cond(is_training, lambda: ty, lambda: vy)

    # network
    net = get_network(net_name, data=data, label=label, )
    y_pred = net.get_output('cls_score')
    decoded = net.get_output('decoded')

    # loss
    dsae_loss = tf.reduce_mean(
        tf.reduce_sum(tf.pow(decoded - tx, 2), axis=1)
    )
    print(dsae_loss)

    cls_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=ty)
    )

    total_loss = dsae_loss + cls_loss

    # train step
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer(hp.lr).minimize(total_loss, global_step=global_step)

    # acc
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(ty, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # sess run
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for epoch in range(hp.num_epochs):
                epoch_loss = np.empty(0)
                for step in range((epoch * tnum_batch), ((epoch + 1) * tnum_batch)):
                    _, tacc, tloss = sess.run(
                        [train_op, accuracy, total_loss],
                        feed_dict={
                            net.keep_prob: hp.dropout,
                            net.mask: 0,
                            is_training: True
                        }
                    )
                    epoch_loss = np.append(epoch_loss, tloss)

                # 每个epoch验证集的acc
                val_acc = sess.run(
                    accuracy,
                    feed_dict={
                        net.keep_prob: hp.dropout,
                        net.mask: 0,
                        is_training: False
                    }
                )
                print('Epoch: ', epoch, 'Train Loss: ', np.mean(epoch_loss), 'Validation Accuracy: ', val_acc)
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train_net('m_2467')
