# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : xgb_train.py
# @Software : PyCharm


import xgboost as xgb
from lib.datasets.tdata import TData
from lib.datasets.data_info import DataInfo as di
from experiment.hyperparams import HyperParams as hp


def train_xgb(mall_name, kidx=0):
    n_feature = di.num_features[mall_name]
    n_class = di.num_classes[mall_name]
    class_map = di.classes[mall_name]
    data = TData(file_path=hp.file_path.format(mall_name), n_class=n_class, n_feature=n_feature, class_map=class_map,
                 kidx=kidx, kfold=hp.kfold)
    tx, ty, vx, vy = data.get_data()

    # XGBoost自带接口
    params = {
        'eta': 0.1,
        'max_depth': 4,
        'min_child_weight': 1,
        'gamma': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'nthread': 4,
        'scale_pos_weight': 1,
        'lambda': 1,
        'seed': 27,
        'silent': 0,
        'eval_metric': 'auc',
        'num_class': n_class,
    }
    d_train = xgb.DMatrix(tx, label=ty)
    d_valid = xgb.DMatrix(vx, label=vy)
    d_test = xgb.DMatrix(vx)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    model_bst = xgb.train(params, d_train, 30, watchlist)
    y_bst = model_bst.predict(d_test)


if __name__ == '__main__':
    # for mall_name in DI.sample20000:
        mall_name = 'm_2467'
        # model = lgb.Booster(model_file=HP.gbm_path.format(mall_name))
        # print(model.dump_model())
        train_xgb(mall_name)
