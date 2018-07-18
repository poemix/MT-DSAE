# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : hybrid_dsae.py
# @Software : PyCharm


import numpy as np
import lightgbm as lgb
import tensorflow as tf
from lib.networks.network import Network
from experiment.hyperparams import HyperParams as hp


class HybridDenosingStackAutoEncoder(Network):
    def __init__(self, data, label, gbm_data, gbm_path):
        self.data = data
        self.gbm_data = gbm_data
        self.label = label
        self.keep_prob = tf.placeholder(tf.float32)

        self.n_class = label.shape[-1]
        self.n_feature = data.shape[-1]

        self.mask = tf.placeholder(tf.float32, [None, self.n_feature])
        self.gbm_path = gbm_path
        self.gbm_model = self.init_gbm(gbm_path)

        super(HybridDenosingStackAutoEncoder, self).__init__(
            inputs={
                'data': self.data,
                'mask': self.mask,
                'label': self.label,
                'keep_prob': self.keep_prob,
            },
            trainable=hp.trainable,
            reuse=hp.reuse
        )

    def init_gbm(self, model_path):
        gbm = lgb.Booster(model_file=model_path)
        return gbm

    def load(self, data_path, sess, saver):
        pass

    def gbm_wrapper(self, gbm_data):
        def func(x):
            y = self.gbm_model.predict(x).astype(np.float32)
            return y, y.shape[0], y.shape[1]

        pred, batch_size, label_size = tf.py_func(func, [gbm_data], [tf.float32, tf.int32, tf.int32])
        pred = tf.reshape(pred, tf.stack([batch_size, self.n_class]))
        return pred

    def label2vec(self, input, label_size, embedding_size, name, initializer, trainable=True, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            label_embeddings = self.make_var('label_embeddings', [label_size, embedding_size], initializer=initializer,
                                             trainable=trainable)
            output = tf.nn.embedding_lookup(label_embeddings, input)
            return output

    def setup(self):

        # denosing stack autoencoder
        data_denosing = tf.multiply(self.data, self.mask)
        enc1 = self.fc(data_denosing, hp.hidden_units512, name='enc1', biased=True, activation='elu')
        enc2 = self.fc(enc1, hp.hidden_units256, name='enc2', biased=True, activation='elu')
        encoded = self.fc(enc2, hp.hidden_units128, name='encoded', biased=True, activation='elu')
        dec2 = self.fc(encoded, hp.hidden_units256, name='dec2', biased=True, activation='elu')
        dec1 = self.fc(dec2, hp.hidden_units512, name='dec1', biased=True, activation='elu')
        decoded = self.fc(dec1, self.n_feature, name='decoded', biased=True, activation=False)

        # gbm
        gbm_pred = self.gbm_wrapper(self.gbm_data)
        print(gbm_pred)  # shape=(?, n_class)

        # fusion
        fusion = tf.concat([encoded, gbm_pred], axis=1, name='fusion')
        # print(gbm_pred)
        # indices = tf.argmax(gbm_pred, axis=1)
        # print(indices)
        #
        # # label embedding
        # label_embedding = self.label2vec(indices, label_size=self.n_class, embedding_size=16, name='embedding',
        #                                  initializer=tf.random_normal_initializer(stddev=0.01))
        #
        # print(encoded)  # shape=(?, 128)
        # print(label_embedding)  # shape=(?, 16)

        # # fusion
        # fusion = tf.concat([encoded, label_embedding], axis=1, name='fusion')
        print(fusion)

        # cls
        fc1 = self.fc(fusion, hp.hidden_units256, name='fc1', biased=True, activation='elu')
        fc2 = self.fc(fc1, hp.hidden_units256, name='fc2', biased=True, activation='elu')
        cls_score = self.fc(fc2, self.n_class, name='cls_score', biased=True, activation='elu')
        cls_prob = tf.nn.softmax(cls_score, name='cls_score')

        self.layers['cls_score'] = cls_score
        self.layers['cls_prob'] = cls_prob
        self.layers['decoded'] = decoded


if __name__ == '__main__':
    from lib.datasets.data_info import DataInfo as di
    from lib.datasets.tdata import TData

    # data info
    mall_name = 'm_2467'
    file_path = 'E:/WiFi_Positioning/data/processed/{}.csv'.format(mall_name)
    gbm_path = hp.gbm_path.format(mall_name)
    n_class = di.num_classes[mall_name]
    n_feature = di.num_features[mall_name]
    class_map = di.classes[mall_name]

    tdata = TData(file_path=file_path, n_feature=n_feature, n_class=n_class, class_map=class_map, kfold=5)
    tx, ty, vx, vy, tgbmx, tgbmy, vgbmx, vgbmy = tdata.get_batch_data()

    net = HybridDenosingStackAutoEncoder(tx, ty, tgbmx, gbm_path)

    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     for i in range(1):
    #         print(i, sess.run(net.gbm_wrapper(tgbmx)))
    #     coord.request_stop()
    #     coord.join(threads)
