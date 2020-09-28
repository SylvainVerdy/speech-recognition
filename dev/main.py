"""
Developper: Sylvain Verdy
contact: sylvain.verdy.pro@gmail.com
"""

import sys
import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import time

import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.optim import optimizer

sys.path.append("../")
from utils.loading import load_data
from models.model import Net
from torch.utils.data import DataLoader

with open("../data/configuration.json", "r") as file:
    data = json.load(file)
data = json.loads(json.dumps(data))
path = data['data']['paths']["train"]
type_ = "train/audio"
BATCH_SIZE = 32
N_EPOCHS = 5
N_classes = 30
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(BATCH_SIZE, 1, 128, 1).to(device)

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)


def train(epoch, data):
    net.train().to(device)
    # zero the parameter gradients
    optimizer.zero_grad()
    inputs, labels = data
    # print(type(inputs))
    inputs = torch.from_numpy(np.asarray(inputs).astype(np.float32))
    permutation = torch.randperm(inputs.size()[0])
    running_loss = 0
    # print(inputs.size()[0])
    count = 0
    batch_losses = []
    for batch_idx in range(0, inputs.size()[0], BATCH_SIZE):
        t0 = time.time()
        count += 1
        optimizer.zero_grad()
        indices = permutation[batch_idx:batch_idx + BATCH_SIZE]
        batch_x, batch_y = inputs[indices], labels[indices]
        #  print(batch_x.shape)
        batch_x = batch_x.reshape(batch_x.size()[0], 1, batch_x.size()[1], batch_x.size()[2])
        # print("###### ", batch_x.shape)
        outputs = net(batch_x.to(device)).to(device)
        #  print(outputs.shape)
        loss = criterion(outputs.to(device), batch_y.to(device))
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        batch_losses.append(loss.item())

        sys.stdout.write('\r')
        sys.stdout.write(" Train data epoch %d [%-100s] %d/%d \t Loss:%f" % (
            epoch, '=' * int((batch_idx / inputs.size()[0]) * 100), batch_idx, inputs.size()[0], loss.item()))
        sys.stdout.flush()
        time.sleep(0.25)
        if batch_idx % inputs.size()[0] == 0:
            test_output = net(batch_x.to(device)).to(device)
            pred_y = torch.argmax(test_output, dim=1)
            print(float(np.array([(x == y) for x, y in zip(batch_y, pred_y)]).astype(int).sum()) / float(
                batch_y.size()[0]))
            accuracy = float(np.array([(x == y) for x, y in zip(batch_y, pred_y)]).astype(int).sum()) / float(
                batch_y.size()[0])
            print("numerateur:", float(np.array([(x == y) for x, y in zip(batch_y, pred_y)]).astype(int).sum()))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                  '| train accuracy: %.2f' % accuracy)

    print("\n")
    print('Epoch {}, loss {}, took {} seconds'.format(epoch, loss.item(), time.time() - t0))
    print("\n")


def main():
    X_train, y_train = load_data(path)
    labels = np.unique(y_train).tolist()
    le = preprocessing.LabelEncoder()
    le.fit_transform(labels)
    y_train = torch.as_tensor(le.transform(y_train))
    X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=40, shuffle=True)

    data = [X_train_, y_train_]
    for epoch in range(N_EPOCHS):
        train(epoch, data)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "../models/model_CNN/model.pt")
    return 0


if __name__ == "__main__":
    main()
