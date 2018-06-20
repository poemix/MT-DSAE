# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : tianchi_data.py
# @Software : PyCharm


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lib.data_layer import data_info


seed = 2018


class TianchiDataLayer(object):
    def __init__(self, data_path, m_name, n_class, n_feature,
                 phase_train=True, triplet=False,
                 cross_validation=True, kfold=5, kidx=0):
        self.data_path = data_path
        self.m_name = m_name
        self.n_class = n_class
        self.n_feature = n_feature

        self.phase_train = phase_train
        self.triplet = triplet
        self.cross_validation = cross_validation
        self.kfold = kfold
        self.kidx = kidx

        if self.triplet or self.cross_validation:
            assert self.phase_train is True

        self.db = self.get_db()

        if self.phase_train:
            self.perm4train = None
            self.idx4train = None
            self.shuffle_db_inds()

            if self.cross_validation:
                self.idx4val = 0
                self.perm4val = np.arange(self.db['n_val_sample'])
        else:
            self.idx4test = 0
            self.perm4test = np.arange(self.db['n_test_sample'])

    def shuffle_db_inds(self):
        self.perm4train = np.random.permutation(np.arange(self.db['n_train_sample']))
        self.idx4train = 0

    def get_db(self):
        df = pd.read_csv('{}\\{}.csv'.format(self.data_path, self.m_name))
        df.drop(
            columns=['latitude', 'longitude', 'time_stamp', 'user_id'],
            axis=1,
            inplace=True
        )

        label = df['shop_id']
        ufunc = np.frompyfunc(lambda x: np.array(data_info.classes[self.m_name][x], np.float32), 1, 1)
        label = np.hstack(ufunc(label))

        df.drop(
            columns=['shop_id'],
            axis=1,
            inplace=True
        )

        df.fillna(-105, inplace=True)
        data = df.values

        # self.mean = data.mean()
        # self.std = data.std()

        # preprocess data and label
        data -= data.mean()
        data /= data.std()
        print(data)
        label = self.preprocess_label(label, len(label), self.n_class)

        db = {
            'n_class': self.n_class,
            'class': np.arange(0, self.n_class),
        }

        if self.phase_train:
            if self.cross_validation:
                kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=seed)
                train_datas = []
                val_datas = []
                train_labels = []
                val_labels = []
                for train_idx, val_idx in kf.split(data, np.argmax(label, axis=1)):
                    train_datas.append(data[train_idx])
                    val_datas.append(data[val_idx])

                    train_labels.append(label[train_idx])
                    val_labels.append(label[val_idx])

                train_data = train_datas[self.kidx]
                val_data = val_datas[self.kidx]
                train_label = train_labels[self.kidx]
                val_label = val_labels[self.kidx]

                db['val_data'] = val_data
                db['n_val_sample'] = len(val_data)
                db['val_label'] = val_label

                print(len(train_data), 'training data.')
                print(len(val_data), 'validation data.')
            else:
                train_label = label
                train_data = data

                print(len(train_data), 'training data.')
            db['train_data'] = train_data
            db['n_train_sample'] = len(train_data)
            db['train_label'] = train_label
            # print(len(train_data))
        else:
            assert self.triplet is False
            assert self.cross_validation is False
            # for test data
            n_sample = len(data)
            db['test_label'] = label

            print(n_sample, 'testing data.')
            db['test_data'] = data
            db['n_test_sample'] = n_sample

        return db

    def preprocess_data(self, data):
        data -= data.mean()
        data /= data.std()
        return data

    def preprocess_label(self, label, n_sample, n_class):
        Y = np.zeros((n_sample, n_class), dtype=np.float32)
        for i, value in enumerate(label):
            Y[i, int(value)] = 1.
        return Y

    def get_next_minibatch_inds4train(self, batch_size):
        if self.idx4train + batch_size >= self.db['n_train_sample']:
            self.shuffle_db_inds()

        db_inds = self.perm4train[self.idx4train: self.idx4train + batch_size]
        self.idx4train += batch_size
        return db_inds

    def get_next_minibatch4train(self, batch_size):
        db_inds = self.get_next_minibatch_inds4train(batch_size)
        blobs = {}
        data_blob = np.zeros((batch_size, self.n_feature), dtype=np.float32)
        label_blob = np.zeros((batch_size, self.n_class), dtype=np.float32)
        for i, idx in enumerate(db_inds):
            data_blob[i] = self.db['train_data'][idx]
            label_blob[i] = self.db['train_label'][idx]
        blobs['data'] = data_blob
        blobs['label'] = label_blob

        return blobs

    def get_next_minibatch_ids4val(self, batch_size):
        assert self.idx4val < self.db['n_val_sample']  # 结束
        db_inds = self.perm4val[self.idx4val: self.idx4val + batch_size]
        self.idx4val += batch_size
        if self.idx4val >= self.db['n_val_sample']:
            self.idx4val = self.db['n_val_sample']
        return db_inds

    def get_next_minibatch4val(self, batch_size):
        db_inds = self.get_next_minibatch_ids4val(batch_size)
        blobs = {}
        data_blob = np.zeros((batch_size, self.n_feature), dtype=np.float32)
        label_blob = np.zeros((batch_size, self.n_class), dtype=np.float32)

        for i, idx in enumerate(db_inds):
            data_blob[i] = self.db['val_data'][idx]
            label_blob[i] = self.db['val_label'][idx]
        blobs['data'] = data_blob
        blobs['label'] = label_blob

        return blobs

    def get_next_minibatch_ids4test(self, batch_size):
        assert self.idx4test <= self.db['n_test_sample']
        db_inds = self.perm4test[self.idx4test: self.idx4test + batch_size]
        self.idx4test += batch_size
        if self.idx4test >= self.db['n_test_sample']:
            self.idx4test = self.db['n_test_sample']
        return db_inds

    def get_next_minibatch4test(self, batch_size):
        db_inds = self.get_next_minibatch_ids4test(batch_size)
        blobs = {}
        data_blob = np.zeros((batch_size, self.n_feature), dtype=np.float32)
        label_blob = np.zeros((batch_size, self.n_class), dtype=np.float32)

        for i, idx in enumerate(db_inds):
            data_blob[i] = self.db['test_data'][idx]
            label_blob[i] = self.db['test_label'][idx]
        blobs['data'] = data_blob
        blobs['label'] = label_blob

        return blobs

    def forward4train(self, batch_size=10):
        blobs = self.get_next_minibatch4train(batch_size)
        return blobs

    def forward4val(self, batch_size=10):
        if self.cross_validation:
            assert batch_size <= self.db['n_val_sample']
            if (self.idx4val + batch_size) <= self.db['n_val_sample']:
                return self.get_next_minibatch4val(batch_size)
            else:
                if self.idx4val >= self.db['n_val_sample']:
                    raise ValueError('not forward.')
                else:
                    return self.get_next_minibatch4val(self.db['n_val_sample'] - self.idx4val)
        else:
            raise ValueError('Cross-validation must be True.')

    def forward4test(self, batch_size=10):
        if not self.phase_train:
            assert batch_size <= self.db['n_test_sample']
            if (self.idx4test + batch_size) <= self.db['n_test_sample']:
                return self.get_next_minibatch4test(batch_size)
            else:
                if self.idx4test >= self.db['n_test_sample']:
                    raise ValueError('not forward.')
                else:
                    return self.get_next_minibatch4test(self.db['n_test_sample'] - self.idx4test)
        else:
            raise ValueError('phase_train must be False.')


if __name__ == '__main__':
    pass
    # mall_name = 'm_4828'
    # data_layer = TianchiDataLayer(data_path='E:\\python_pro\\wifi\\dataset\\tianchi',
    #                               m_name=mall_name, n_class=num_classes[mall_name],
    #                               n_feature=num_features[mall_name])
    # print(data_layer.db['train_data'].shape)
    # ii = data_layer.forward4train(5)
    # print(ii['data'])
    # print(ii['label'])
    # jj = data_layer.forward4val(5)
    # print(jj['data'])
    # print(jj['label'])
