# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : factory.py
# @Software : PyCharm


from lib.networks.sae import StackAutoEncoder as sae
from lib.networks.dynamic_sae import StackAutoEncoder as dynamic_sae
from lib.networks.nn import NeuralNetwork as nn
from lib.networks.denosing_sae import DenosingStackAutoEncoder as denosing_sae
from lib.networks.cnn_sae import CNNStackAutoEncoder as cnn_sae


def get_network(name, **kwargs):
    n_feature = kwargs.get('n_feature', None)
    n_class = kwargs.get('n_class', None)
    trainable = kwargs.get('trainable', True)
    reuse = kwargs.get('reuse', False)
    if name == 'sae':
        return sae(n_class=n_class, trainable=trainable, n_feature=n_feature, reuse=reuse)
    elif name == 'dynamic_sae':
        return dynamic_sae(n_class=n_class, trainable=trainable, n_feature=n_feature, reuse=reuse)
    elif name == 'nn':
        return nn(n_class=n_class, trainable=trainable, n_feature=n_feature, reuse=reuse)
    elif name == 'denosing_sae':
        return denosing_sae(n_class=n_class, trainable=trainable, n_feature=n_feature, reuse=reuse)
    elif name == 'cnn_sae':
        return cnn_sae(n_class=n_class, trainable=trainable, n_feature=n_feature, reuse=reuse)
    else:
        raise KeyError('Unknown dataset: {}'.format(name))
