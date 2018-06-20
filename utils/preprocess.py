# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : preprocess.py
# @Software : PyCharm


import pandas as pd

path = 'E:\\python_pro\\competition\\tianchi\\shop\\dataset\\preliminary\\'

df = pd.read_csv(path + 'ccf_first_round_user_shop_behavior.csv')
shop = pd.read_csv(path + 'ccf_first_round_shop_info.csv')
test = pd.read_csv(path + 'evaluation_public.csv')
df = pd.merge(df, shop[['shop_id', 'mall_id']], how='left', on='shop_id')
df['time_stamp'] = pd.to_datetime(df['time_stamp'])
train = pd.concat([df, test])
mall_list = list(set(list(shop.mall_id)))
result = pd.DataFrame()
for mall in mall_list:
    train1 = train[train.mall_id == mall].reset_index(drop=True)
    l = []
    wifi_dict = {}
    for index, row in train1.iterrows():
        r = {}
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            r[i[0]] = int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]] = 1
            else:
                wifi_dict[i[0]] += 1
        l.append(r)
    delate_wifi = []
    for i in wifi_dict:
        if wifi_dict[i] < 20:
            delate_wifi.append(i)
    m = []
    for row in l:
        new = {}
        for n in row.keys():
            if n not in delate_wifi:
                new[n] = row[n]
        m.append(new)
    train1 = pd.concat([train1, pd.DataFrame(m)], axis=1)
    df_train = train1[train1.shop_id.notnull()]
    df_test = train1[train1.shop_id.isnull()]
    df_train.drop(['wifi_infos', 'row_id', 'mall_id'], axis=1, inplace=True)
    print(df_train)
    df_train.to_csv('{}.csv'.format(mall), index=False)
    # break
