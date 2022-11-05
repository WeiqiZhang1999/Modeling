# -*- coding: UTF-8 -*-
"""
@Project ：modeling_baseline 
@File    ：test.py.py
@Author  ：Weiqi Zhang
@Date    ：11/3/2022 1:12 PM 
"""
import pandas as pd
import torch
from model import get_model

data_path = './dataset/trainX.xlsx'
label_path = './dataset/trainY.xlsx'
test_path = './dataset/testX.xlsx'

train_data = pd.read_excel(data_path, index_col='jieju_cust_num')
label = pd.read_excel(label_path)
test_data = pd.read_excel(test_path, index_col='jieju_cust_num')

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

name = all_features.columns
drop_columns = [name[x] for x in range(all_features.shape[1]) if all_features.iloc[:, x].dtype == 'O']
all_features = all_features.drop(columns=drop_columns, axis=0)
all_features = all_features.loc[:, (all_features != all_features.iloc[0]).any()]

# Normalization
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features]
all_features = all_features.fillna(0)

n_train = train_data.shape[0]
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

model = get_model()
model.load_state_dict(torch.load('checkpoints/linear_model.pth'))
print(f"loaded model")

model.eval()

y_test = pd.DataFrame(list(test_data.index), columns=['ID'])

with torch.no_grad():
    test_result = model(test_features)


test_list = test_result.detach().numpy()
for idx in range(len(test_list)):
    if test_list[idx] < 0.5:
        test_list[idx] = 0
    else:
        test_list[idx] = 1

y_test['LABEL'] = test_list
y_test.to_excel('linear_model.xlsx', index=False)