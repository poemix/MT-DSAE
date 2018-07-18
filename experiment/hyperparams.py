# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : hyperparams.py
# @Software : PyCharm


class HyperParams:
    """
    HyperParameters
    """
    # data
    file_path = 'E:/WiFi_Positioning/data/processed/{}.csv'

    gbm_path = 'E:/WiFi_Positioning/models/GBDT/{}.model'
    NN_model_path = 'E:/WiFi_Positioning/models/NN/{}.ckpt'

    # training
    batch_size = 32
    tbatch_size = 15
    vbatch_size = 32
    lr = 0.00001
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
