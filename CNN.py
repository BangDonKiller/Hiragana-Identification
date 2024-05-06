import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001

train_data = pd.read_csv('./dataset/train_data.csv', sep=',')
train_data_label = train_data["label"].values.tolist()
train_data_label = torch.from_numpy(
    np.array(train_data_label)).type(torch.LongTensor)
train_data = train_data.iloc[:, 2:].values/255.
train_data = torch.from_numpy(train_data).float()
train_data = train_data.view(-1, 1, 28, 28)


test_data = pd.read_csv('./dataset/test_data.csv', sep=',')
test_data_label = test_data["label"].values.tolist()
test_data_label = torch.from_numpy(
    np.array(test_data_label)).type(torch.LongTensor)
test_data = test_data.iloc[:, 2:].values/255.
test_data = torch.from_numpy(test_data).float()
test_data = test_data.view(-1, 1, 28, 28)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

cnn = CNN()

optimization = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step in range(train_data.size(0)//BATCH_SIZE):
        b_x = Variable(train_data[step:step+BATCH_SIZE])

        b_y = Variable(train_data_label[step:step+BATCH_SIZE])
        # print("b_x:", b_x, " b_x.size():", b_x.size())
        # print("b_y:", b_y, " b_y.size():", b_y.size())

        output = cnn(b_x)[0]

        loss = loss_func(output, b_y)
        optimization.zero_grad()
        loss.backward()
        optimization.step()

        if step % 50 == 0:
            test_output, last_layer = cnn(test_data)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

            correct = 0
            for i in range(test_data.size(0)):
                if (pred_y[i] == test_data_label[i].numpy()):
                    correct += 1
            accuracy = correct / test_data.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' %
                  loss.data, '| test accuracy: %.2f' % accuracy)

test_output, _ = cnn(test_data[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(test_data_label[:10], 'real number')
print(pred_y, 'prediction Hiragana')
