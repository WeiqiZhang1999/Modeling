# -*- coding: UTF-8 -*-
"""
@Project ：modeling_baseline
@File    ：data_helper.py
@Author  ：Weiqi Zhang
@Date    ：11/3/2022 12:16 AM 
"""

import torch
from torch.utils.data import Dataset
import pandas as pd


class train_dataset(Dataset):
    def __init__(self, data_path, label_path):
        data = pd.read_excel(data_path, index_col='jieju_cust_num')
        label = pd.read_excel(label_path)

        # data preprocessing
        name = data.columns
        drop_columns = [name[x] for x in range(data.shape[1]) if data.iloc[:, x].dtype == 'O']
        train_data = data.drop(columns=drop_columns, axis=0)
        train_data = train_data.loc[:, (train_data != train_data.iloc[0]).any()]

        # Normalization
        numeric_features = train_data.dtypes[data.dtypes != 'object'].index
        train_data[numeric_features] = train_data[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        train_data[numeric_features] = train_data[numeric_features]
        train_data = train_data.fillna(0)
        # data = pd.get_dummies(data, dummy_na=True)

        self.x_data = torch.tensor(train_data.values, dtype=torch.float32)
        self.y_data = torch.tensor(label['jieju_dubil_status_desc'].values, dtype=torch.float32)
        self.train_len = train_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.train_len
