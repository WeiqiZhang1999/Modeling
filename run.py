# -*- coding: UTF-8 -*-
"""
@Project ：modeling_baseline
@File    ：run.py
@Author  ：Weiqi Zhang
@Date    ：11/3/2022 12:00 AM 
"""

import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import get_model
from data_helper import train_dataset


def plot_loss(losses):
    figure = plt.figure(figsize=(9, 3))
    plt.subplot(121, title="losses")
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("epoch no")
    plt.ylabel("loss")
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument("--data_path", type=str, default='./dataset/trainX.xlsx', help="path to training data")
    parser.add_argument("--label_path", type=str, default='./dataset/trainY.xlsx', help="path to training label")
    parser.add_argument("--save_path", type=str, default='./checkpoints', help="path to saving the model")
    # hyper parameters
    parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay of optimizer")
    parser.add_argument("--experiment_name", type=str, default='linear_model', help="experimental name")

    opt = parser.parse_args()

    data_path = opt.data_path
    label_path = opt.label_path
    save_path = opt.save_path
    lr = opt.lr
    weight_decay = opt.weight_decay
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    experiment_name = opt.experiment_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # create model
    model = get_model()
    model.to(device)

    # prepare datasets
    train_data = train_dataset(data_path, label_path)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    # prepare optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # criterion
    criterion = nn.BCELoss(reduction='mean')

    # training loop
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        batches = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_pred = model(inputs)
            y_pred = y_pred.squeeze(-1)
            # print(y_pred)
            # print(labels)
            loss = criterion(y_pred, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1
        losses.append(running_loss / batches)
        print('epoch {} done'.format(epoch + 1))
    print('Finish Traning!')

    plot_loss(losses)
    model_name = experiment_name + '.pth'
    torch.save(model.state_dict(), os.path.join(save_path, model_name))


if __name__ == '__main__':
    main()
