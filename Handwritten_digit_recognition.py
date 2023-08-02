import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # 第一层卷积层， 按照Sequential的结构组织
        # 输出形状(6, 14, 14)
        self.conv_pool1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  # 输入通道1
                      out_channels=6,  # 输出通道6
                      kernel_size=(5, 5),  # 卷积核面积5x5
                      padding=2),   # 步长是2
            nn.ReLU(),  # 设定Relu激活函数
            nn.MaxPool2d(2, stride=2)  # 池化
        )
        # 第二层卷积层， 按照Sequential的结构组织
        # 输出形状(16, 10, 10)
        self.conv_pool2 = nn.Sequential(
            nn.Conv2d(in_channels=6,  # 输入通道6
                      out_channels=16,  # 输出通道16
                      kernel_size=(5, 5)  # 卷积核面积5x5
                      ),
            nn.ReLU(),  # 设定Relu激活函数
            nn.MaxPool2d(2, stride=2)  # 池化  # output(16, 5, 5)
        )

        # 第三层全连接层，120->160
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )

        # 第四层全连接层，160->84
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # 第五层输出层，84->10
        self.out = nn.Sequential(
            nn.Linear(84, 10),
        )

    # 设定网络的前向传播函数
    def forward(self, x):
        x = self.conv_pool1(x)
        x = self.conv_pool2(x)
        x = x.view(x.size(0), -1)  # 展平成1维
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


# 加载数据，使用torch库下载MNIST手写数据集
def load_mydata(batch_size):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='mnist', train=True, transform=transform, download=True)
    train_loaders = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_set = torchvision.datasets.MNIST(root='mnist', train=False, transform=transform, download=True)
    test_loaders = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loaders, test_loaders


# 训练函数
def train(model, learn_rate, train_loaders, epoch):
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)  # 定义优化器
    loss_fun = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    print_batch = 200  # 每固定轮Batch输出一次训练情况的Loss
    for i in range(epoch):
        total_loss = 0.0
        for ii, (train_data, train_label) in enumerate(train_loaders):
            train_data = Variable(train_data, requires_grad=True)
            train_label = Variable(train_label)
            optimizer.zero_grad()  # 清空将前一次的梯度
            out = model(train_data)  # 前向传播
            loss = loss_fun(out, train_label)  # 计算误差
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            total_loss += loss
            if (ii + 1) % print_batch == 0:
                print(f'epoch NO. {i+1}, batch NO. {ii+1} ,recent {print_batch} loss: {total_loss / print_batch}')
                total_loss = 0.0
    print("训练结束")


def test(model, test_loaders):
    correct = 0
    total = 0
    for datas in test_loaders:
        images, labels = datas
        images = Variable(images)
        outputs = model(images)
        predicted = torch.argmax(outputs.data, dim=-1)
        total += labels.size(0)
        correct += sum(predicted == labels)
    print('预测准确率 %.2f %%' % (100 * correct / total))


def show(model, test_loaders):
    for datas in test_loaders:
        images, labels = datas
        images = Variable(images)
        outputs = model(images)
        for i in range(images.shape[0]):
            img = images[i][0].numpy()
            real = labels[i]
            a = str(np.argmax(outputs[i].detach().numpy()))
            b = str(real.detach().numpy())
            info = "Predict " + a + " Real " + b
            img = cv2.resize(img, (400, 400))
            cv2.imshow(info, img)
            print(info)
            cv2.waitKey(0)


def main():
    model = Model()  # 加载网络
    batch_size = 32  # 定义Batch Size为32，每次抓取32张图片
    learn_rate = 0.001  # 定义网络学习率为0.001
    epoch = 1  # 定义学习轮次
    train_loader, test_loader = load_mydata(batch_size)  # 加载数据集和训练集
    train(model, learn_rate, train_loader, epoch)  # 使用训练集数据训练网络
    test(model, test_loader)  # 使用测试集数据测试网络，并输出网络评价分数
    show(model, test_loader)  # 展示测试集的图片和标签与网络预测的标签


main()
