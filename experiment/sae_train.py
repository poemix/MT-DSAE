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
    # tf.set_random_seed(hp.seed)
    n_class = di.num_classes[mall_name]
    n_feature = di.num_features[mall_name]
    class_map = di.classes[mall_name]
    file_path = hp.file_path.format(mall_name)
    print(n_class, n_feature)

    # data
    tdata = TData(file_path=file_path, n_feature=n_feature, n_class=n_class, class_map=class_map,
                  kfold=hp.kfold)
    with tf.device("/cpu:0"):
        tx, ty, n_tsample, vx, vy, n_esample = tdata.get_batch_data(kidx=0)

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
    loss = tf.add(sae_loss, cls_loss)

    # train step
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer(hp.learning_rate).minimize(loss, global_step=global_step)

    # acc
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(net.label, 1))
    num_correct_pred = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # sess run
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for epoch in range(hp.num_epochs):
                epoch_loss = np.empty(0)
                num_tbatch = n_tsample // hp.tbatch_size
                tdata, tlabel = sess.run([tx, ty])
                # print(tdata)
                for step in range((epoch * num_tbatch), ((epoch + 1) * num_tbatch)):
                    _, tacc, tloss = sess.run(
                        [train_op, accuracy, loss],
                        feed_dict={
                            net.data: tdata,
                            net.label: tlabel,
                            net.keep_prob: hp.dropout,
                            net.is_training: True
                        }
                    )
                    epoch_loss = np.append(epoch_loss, tloss)

                ebs = 100
                num_ebatch = n_esample // ebs + 1
                cnt = 0
                for estep in range(num_ebatch):
                    # 每个epoch验证集的acc
                    if estep == num_ebatch - 1:
                        ebs = n_esample % ebs
                    vdata, vlabel = sess.run([vx, vy], feed_dict={hp.ebatch_size: ebs})
                    num = sess.run(
                        num_correct_pred,
                        feed_dict={
                            net.data: vdata,
                            net.label: vlabel,
                            net.keep_prob: 1.,
                            net.is_training: False
                        }
                    )
                    cnt += num
                print('Epoch: ', epoch, 'Train Loss: ', np.mean(epoch_loss), 'Validation Accuracy: ', cnt / n_esample)
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train_net('m_2467')
