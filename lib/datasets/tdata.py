# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : tdata.py
# @Software : PyCharm


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from lib.datasets.data_info import DataInfo as di
from experiment.hyperparams import HyperParams as hp


class TData:
    def __init__(self, file_path, n_class, n_feature, class_map,
                 kfold=-1, fillna=-105, shuffle=True):
        self.file_path = file_path
        self.n_class = n_class
        self.n_feature = n_feature
        self.class_map = class_map  # str map to 0, 1, 2, ....
        self.kfold = kfold
        self.fillna = fillna
        self.shuffle = shuffle

        # init
        self.data = None
        self.label = None

        if self.kfold >= 2:
            self.tidxs = []
            self.eidxs = []

        self.init(file_path)

    def init(self, data_path):
        df = pd.read_csv(data_path)
        if self.shuffle:
            df.sample(frac=1, random_state=hp.seed).reset_index(drop=True)

        label = df['shop_id']
        func = np.frompyfunc(lambda x: np.array(self.class_map[x], np.float32), 1, 1)
        self.label = np.hstack(func(label))

        df.drop(
            columns=['latitude', 'longitude', 'time_stamp', 'user_id', 'shop_id'],
            axis=1,
            inplace=True
        )

        df.fillna(self.fillna, inplace=True)
        self.data = df.values

        if self.kfold >= 2:
            kf = StratifiedKFold(n_splits=self.kfold, shuffle=self.shuffle, random_state=hp.seed)
            for tidx, eidx in kf.split(self.data, self.label):
                self.tidxs.append(tidx)
                self.eidxs.append(eidx)

    def get_data(self, kidx=0):
        """
        
        :param kidx: 
        :return: 
        """
        if self.kfold >= 2:
            tx = self.data[self.tidxs[kidx]]
            ty = self.label[self.tidxs[kidx]]

            ex = self.data[self.eidxs[kidx]]
            ey = self.label[self.eidxs[kidx]]

            return tx, ty, ex, ey
        else:
            return self.data, self.label

    def get_batch_data(self, kidx=0):
        """
        data for nn training
        :return: 
        """

        def process_label(label, n_sample, n_class):
            y = np.zeros((n_sample, n_class), dtype=np.float32)
            for i, value in enumerate(label):
                y[i, int(value)] = 1.
            return y

        def process_data(data):
            data -= data.mean()
            data /= data.std()
            return data

        # pre-process data and label
        X = process_data(self.data.copy())
        Y = process_label(self.label, len(self.label), self.n_class)

        # convert to tensor
        X = tf.convert_to_tensor(X, tf.float32)
        Y = tf.convert_to_tensor(Y, tf.float32)

        if self.kfold >= 2:
            n_tsample = len(self.tidxs[kidx])
            n_esample = len(self.eidxs[kidx])

            TX = tf.gather(X, indices=self.tidxs[kidx])
            TY = tf.gather(Y, indices=self.tidxs[kidx])

            EX = tf.gather(X, indices=self.eidxs[kidx])
            EY = tf.gather(Y, indices=self.eidxs[kidx])

            train_input_queues = tf.train.slice_input_producer([TX, TY], shuffle=False)
            eval_input_queues = tf.train.slice_input_producer([EX, EY], shuffle=False)

            # create batch queues
            tx, ty = tf.train.shuffle_batch(train_input_queues, num_threads=2, batch_size=hp.tbatch_size,
                                            capacity=1000, min_after_dequeue=500)
            ex, ey = tf.train.batch(eval_input_queues, num_threads=2, batch_size=hp.ebatch_size,
                                    capacity=1000)
            return tx, ty, n_tsample, ex, ey, n_esample
        else:
            # calc total batch count
            n_sample = len(X)

            # Create Queues
            input_queues = tf.train.slice_input_producer([X, Y])

            # create batch queues
            x, y = tf.train.shuffle_batch(input_queues, num_threads=2, batch_size=hp.tbatch_size,
                                          capacity=1000, min_after_dequeue=500)
            return x, y, n_sample

            # def get_batch_data_v2(self):
            #     """
            #     data for nn training
            #     :return:
            #     """
            #
            #     def process_label(label, n_sample, n_class):
            #         y = np.zeros((n_sample, n_class), dtype=np.float32)
            #         for i, value in enumerate(label):
            #             y[i, int(value)] = 1.
            #         return y
            #
            #     def process_data(data):
            #         data -= data.mean()
            #         data /= data.std()
            #         return data
            #
            #     # preprocess data and label
            #     X = process_data(self.data.copy())
            #     Y = process_label(self.label, len(self.label), self.n_class)
            #
            #     # calc total batch count
            #     num_batch = len(X) // hp.batch_size
            #
            #     # Convert to tensor
            #     X = tf.convert_to_tensor(X, tf.float32)
            #     Y = tf.convert_to_tensor(Y, tf.float32)
            #
            #     GBMX = tf.convert_to_tensor(self.data, tf.float32)
            #     GBMY = tf.convert_to_tensor(self.label, tf.float32)
            #
            #     if self.kfold >= 2:
            #         tnum_batch = len(self.tidxs[self.kidx]) // hp.tbatch_size
            #         vnum_batch = len(self.vidxs[self.kidx]) // hp.ebatch_size
            #         TX = tf.gather(X, indices=self.tidxs[self.kidx])
            #         TY = tf.gather(Y, indices=self.tidxs[self.kidx])
            #
            #         VX = tf.gather(X, indices=self.vidxs[self.kidx])
            #         VY = tf.gather(Y, indices=self.vidxs[self.kidx])
            #
            #         TGBMX = tf.gather(GBMX, indices=self.tidxs[self.kidx])
            #         TGBMY = tf.gather(GBMY, indices=self.tidxs[self.kidx])
            #
            #         VGBMX = tf.gather(GBMX, indices=self.vidxs[self.kidx])
            #         VGBMY = tf.gather(GBMY, indices=self.vidxs[self.kidx])
            #
            #         train_input_queues = tf.train.slice_input_producer([TX, TY, TGBMX, TGBMY])
            #         val_input_queues = tf.train.slice_input_producer([VX, VY, VGBMX, VGBMY])
            #
            #         # create batch queues
            #         tx, ty, tgbmx, tgbmy = tf.train.shuffle_batch(train_input_queues, num_threads=8, batch_size=hp.batch_size,
            #                                                       capacity=hp.batch_size * 64,
            #                                                       min_after_dequeue=hp.batch_size * 32,
            #                                                       allow_smaller_final_batch=False)
            #         vx, vy, vgbmx, vgbmy = tf.train.shuffle_batch(val_input_queues, num_threads=8, batch_size=hp.batch_size,
            #                                                       capacity=1000,
            #                                                       min_after_dequeue=500,
            #                                                       allow_smaller_final_batch=False)
            #         return tx, ty, vx, vy, tgbmx, tgbmy, vgbmx, vgbmy, tnum_batch, vnum_batch
            #     else:
            #         # Create Queues
            #         input_queues = tf.train.slice_input_producer([X, Y, GBMX, GBMY])
            #
            #         # create batch queues
            #         x, y, gbmx, gbmy = tf.train.shuffle_batch(input_queues, num_threads=8, batch_size=hp.tbatch_size,
            #                                                   capacity=1000, min_after_dequeue=500)
            #         return x, y, gbmx, gbmy, num_batch


if __name__ == '__main__':
    # data info
    mall_name = 'm_2467'
    file_path = 'E:/WiFiPositioning/data/processed/{}.csv'.format(mall_name)
    n_class = di.num_classes[mall_name]
    n_feature = di.num_features[mall_name]
    class_map = di.classes[mall_name]

    data = TData(file_path=file_path, n_feature=n_feature, n_class=n_class, class_map=class_map, kfold=5)
    with tf.device('/cpu:0'):
        tx, ty, n_tsample, ex, ey, n_esample = data.get_batch_data()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        batch_size = 32
        num_batch = n_esample // batch_size + 1
        for step in range(num_batch):
            if step == num_batch - 1:
                batch_size = n_esample % batch_size
            k = sess.run(
                [ex, ey],
                feed_dict={
                    hp.ebatch_size: batch_size
                }
            )
            print(step, k[0].shape)
        coord.request_stop()
        coord.join(threads)

        print(divmod(n_esample, 32))
