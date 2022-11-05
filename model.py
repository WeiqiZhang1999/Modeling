# -*- coding: UTF-8 -*-
"""
@Project ：modeling_baseline
@File    ：model.py
@Author  ：Weiqi Zhang
@Date    ：11/1/2022 5:24 PM 
"""

import torch.nn as nn
import torch.nn.functional as F


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.fc1 = nn.Linear(106, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


