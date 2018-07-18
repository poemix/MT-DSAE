# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : sae_train.py
# @Software : PyCharm


import numpy as np
import tensorflow as tf
from lib.datasets.tdata import TData
from lib.datasets.data_info import DataInfo as di
from lib.networks.factory import get_network
from experiment.hyperparams import HyperParams as hp


def train_net(mall_name, net_name='sae', kidx=0):
    """
    train network
    :param mall_name: mall name
    :param net_name: network name
    :param kidx: k'th k-flod cross validation
    :return: 
    """
    tf.set_random_seed(hp.seed)
    n_class = di.num_classes[mall_name]
    n_feature = di.num_features[mall_name]
    class_map = di.classes[mall_name]
    file_path = hp.file_path.format(mall_name)
    print(n_class, n_feature)

    # data
    tdata = TData(file_path=file_path, n_feature=n_feature, n_class=n_class, class_map=class_map, kidx=kidx,
                  kfold=hp.kfold)
    with tf.device("/cpu:0"):
        tx, ty, vx, vy, tnum_batch, vnum_batch = tdata.get_batch_data()

    x = tf.placeholder(tf.float32, [None, n_feature])
    y = tf.placeholder(tf.float32, [None, n_class])

    # network
    net = get_network(net_name, data=x, label=y)
    y_pred = net.get_output('cls_score')
    decoded = net.get_output('decoded')

    # loss
    sae_loss = tf.reduce_mean(tf.pow(decoded - net.data, 2))
    # print(sae_loss)

    cls_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=net.label)
    )
    total_loss = sae_loss + cls_loss

    # train step
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer(hp.lr).minimize(total_loss, global_step=global_step)

    # acc
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(net.label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # sess run
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for epoch in range(hp.num_epochs):
                epoch_loss = np.empty(0)
                tdata, tlabel = sess.run([tx, ty])
                for step in range((epoch * tnum_batch), ((epoch + 1) * tnum_batch)):
                    _, tacc, tloss = sess.run(
                        [train_op, accuracy, total_loss],
                        feed_dict={
                            net.data: tdata,
                            net.label: tlabel,
                            net.keep_prob: hp.dropout,
                            net.is_training: True
                        }
                    )
                    epoch_loss = np.append(epoch_loss, tloss)

                # 每个epoch验证集的acc
                vdata, vlabel = sess.run([vx, vy])
                val_acc = sess.run(
                    accuracy,
                    feed_dict={
                        net.data: vdata,
                        net.label: vlabel,
                        net.keep_prob: 1.,
                        net.is_training: False
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
