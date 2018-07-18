# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : factory.py
# @Software : PyCharm


from lib.networks.sae import StackAutoEncoder
from lib.networks.dnn import DeepNeuralNetwork
from lib.networks.dsae import DenosingStackAutoEncoder


def get_network(name, data, label):
    if name == 'dnn':
        net = DeepNeuralNetwork(data, label)
    elif name == 'sae':
        net = StackAutoEncoder(data, label)
    elif name == 'dsae':
        net = DenosingStackAutoEncoder(data, label)
    else:
        raise KeyError('Unknown Network: {}'.format(name))

    return net


if __name__ == '__main__':
    import tensorflow as tf
    from lib.datasets.tdata import TData
    from lib.datasets.data_info import DataInfo as di
    from experiment.hyperparams import HyperParams as hp

    # data info
    mall_name = 'm_2467'
    file_path = 'E:/WiFi_Positioning/data/processed/{}.csv'.format(mall_name)
    gbm_path = hp.gbm_path.format(mall_name)
    n_class = di.num_classes[mall_name]
    n_feature = di.num_features[mall_name]
    class_map = di.classes[mall_name]

    tdata = TData(file_path=file_path, n_feature=n_feature, n_class=n_class, class_map=class_map, kfold=5)
    tx, ty, vx, vy, tnum_batch, vnum_batch = tdata.get_batch_data()
    net = get_network('sae', tx, ty)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1):
            print(i, sess.run(net.get_output('cls_prob')))
        coord.request_stop()
        coord.join(threads)
