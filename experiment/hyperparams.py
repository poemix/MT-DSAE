# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : hyperparams.py
# @Software : PyCharm


import tensorflow as tf


class HyperParams:
    """
    HyperParameters
    """
    # data
    file_path = 'E:/WiFiPositioning/data/processed/{}.csv'

    gbm_path = 'E:/WiFiPositioning/models/GBDT/{}.model'
    NN_model_path = 'E:/WiFiPositioning/models/NN/{}.ckpt'

    # training
    tbatch_size = 16  # for train
    ebatch_size = tf.placeholder(tf.int32)  # for eval
    learning_rate = 0.00001
    logdir = 'logdir'

    # network param
    hidden_units512 = 512
    hidden_units256 = 256
    hidden_units128 = 128
    hidden_units64 = 64
    num_epochs = 50
    dropout = 0.8

    trainable = True
    reuse = None

    kfold = 5

    activation = 'elu'
    seed = 2018

    # data param
    fill_na = -105
