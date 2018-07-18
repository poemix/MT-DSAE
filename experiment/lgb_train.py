# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : lgb_train.py
# @Software : PyCharm


import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from lib.datasets.tdata import TData
from lib.datasets.data_info import DataInfo as di
from experiment.hyperparams import HyperParams as hp


def train_gbdt(mall_name, max_iter=32, kidx=0):
    n_feature = di.num_features[mall_name]
    n_class = di.num_classes[mall_name]
    class_map = di.classes[mall_name]
    data = TData(file_path=hp.file_path.format(mall_name), n_class=n_class, n_feature=n_feature, class_map=class_map,
                 kidx=kidx, kfold=hp.kfold)
    tx, ty, vx, vy = data.get_data()

    lgb_train = lgb.Dataset(tx, ty)
    lgb_eval = lgb.Dataset(vx, vy)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        # 'max_depth': 4,
        # 'num_trees': 32,
        'num_leaves': 32,
        'bagging_fraction': 0.9,
        'feature_fraction': 0.8,
        'bagging_freq': 5,
        'objective': 'multiclass',
        # 'metric': 'multi_error',
        'num_class': n_class,
        # 'lambda_l1': 0.4,
        # 'lambda_l2': 0.5,
        'seed': 2018,
        'verbose': -1,
    }

    gbm = lgb.train(params, lgb_train, num_boost_round=max_iter, valid_sets=lgb_eval)
    gbm.save_model(hp.gbm_path.format(mall_name))
    y_pred = gbm.predict(vx)
    print(accuracy_score(vy, np.argmax(y_pred, axis=1)))


if __name__ == '__main__':
    # for mall_name in di.sample20000:
    mall_name = 'm_2467'
    model = lgb.Booster(model_file=hp.gbm_path.format(mall_name))
    dump = model.dump_model()
    tree_info = dump['tree_info']
    tree_counts = len(dump['tree_info'])
    # print(dump['num_tree_per_iteration'])
    # print(dump['max_feature_idx'])
    # print(dump['label_index'])
    # print(dump['version'])
    # print(dump['num_class'])
    # print(dump['name'])
    # print(dump['feature_names'])
    # print(len(dump['tree_info']))
    # print(dump.keys())
    # for i in range(len(dump['tree_info'])):
    #     print(dump['tree_info'][i]['num_leaves'])

    n_feature = di.num_features[mall_name]
    n_class = di.num_classes[mall_name]
    class_map = di.classes[mall_name]
    data = TData(file_path=hp.file_path.format(mall_name), n_class=n_class, n_feature=n_feature, class_map=class_map,
                 kfold=5)
    tx, ty, vx, vy = data.get_data()
    print(vx.shape)

    y_pred = model.predict(vx, pred_leaf=True, num_iteration=model.best_iteration)
    print(y_pred.shape)
    for i in range(tree_counts):
        print(y_pred[0, i])
        # y_pred[:, i] = y_pred[:, i] + dump['tree_info'][i]['num_leaves'] * i + 1
    # print(y_pred)

    train_gbdt(mall_name, max_iter=32)
