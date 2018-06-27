# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Time     : 2018/3/14 14:54
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : data.py
# @Software : PyCharm


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


seed = 2018
np.random.seed(seed)

label_maps = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "10": 4,
    "11": 5,
    "12": 6,
    "13": 7,
    "20": 8,
    "21": 9,
    "22": 10,
    "23": 11,
    "24": 12
}

mean = np.array([-118.56377734093162], dtype=np.float32)
std = np.array([7.940541949050653], dtype=np.float32)


class DataLayer(object):
    def __init__(self, data_path, n_class,
                 phase_train=True, triplet=True,
                 cross_validation=True, kfold=5, kidx=0):
        self.data_path = data_path
        self.n_class = n_class

        self.phase_train = phase_train
        self.triplet = triplet
        self.cross_validation = cross_validation
        self.kfold = kfold
        self.kidx = kidx

        self.n_feature = 520

        if self.triplet or self.cross_validation:
            assert self.phase_train is True

        self.db = self.get_db()

        if self.phase_train:
            # for train
            self.perm4train = None
            self.idx4train = None
            self.shuffle_db_inds4train()

            if self.cross_validation:
                # for val
                self.idx4val = 0
                self.perm4val = np.random.permutation(np.arange(self.db['n_val_sample']))
        else:
            # for test
            self.idx4test = 0
            self.perm4test = np.arange(self.db['n_test_sample'])

    def shuffle_db_inds4train(self):
        """
        Randomly permute the training data
        :return:
        """
        self.perm4train = np.random.permutation(np.arange(self.db['n_train_sample']))
        self.idx4train = 0

    def get_db(self):
        df = pd.read_csv(self.data_path)
        df.drop(
            columns=['LONGITUDE', 'LATITUDE', 'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP'],
            axis=1,
            inplace=True
        )
        for i in range(1, 521):
            j = str(i).zfill(3)
            column = 'WAP{}'.format(j)
            df[column].replace(100, -120, inplace=True)

        label = np.asarray(df['BUILDINGID'].map(str) + df['FLOOR'].map(str), dtype=np.float32)

        ufunc = np.frompyfunc(lambda x: np.array(label_maps[str(x)], np.float32), 1, 1)
        label = np.hstack(ufunc(label.astype(np.int32)))

        df.drop(
            columns=['BUILDINGID', 'FLOOR'],
            axis=1,
            inplace=True
        )

        data = df.values
        data = data.astype(np.float32)

        self.mean = data.mean()
        self.std = data.std()
        self.var = data.var()

        # preprocess data and label
        data = self.preprocess_data(data)
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

    def preprocess_label(self, label, n_sample, n_class):
        Y = np.zeros((n_sample, n_class), dtype=np.float32)
        for i, value in enumerate(label):
            Y[i, int(value)] = 1.
        return Y

    def preprocess_data(self, data):
        # data -= self.mean
        # data /= self.std
        data -= data.min()
        data /= data.max()
        return data

    def get_next_minibatch_inds4train(self, batch_size):
        if self.idx4train + batch_size >= self.db['n_train_sample']:
            self.shuffle_db_inds4train()

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
    data_layer = DataLayer(
        'E:\\python_pro\\zjgsu\\data\\UJIndoorLoc\\trainingData.csv', 13,
        cross_validation=True, phase_train=True, triplet=False
    )
    ii = data_layer.forward4train(10)
    print(ii['data'])
    print(ii['label'])
