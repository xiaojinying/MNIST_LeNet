# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST      #获取MNIST的数据集
from torch.utils.data import DataLoader
import os


# 计算出mnist数据中的均值与标准差，便于进行标准化
# train_dataset = MNIST(root='/MNIST_data', train=True, download=True,
#                       transform=transforms.ToTensor())
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=60000)
# for img, target in train_loader:
#     x = img.view(-1, 28 * 28)
#     print(x.shape)
#     print(x.mean())
#     print(x.std())
#     print(set(target.numpy()))  # 查看标签有多少个类别
#     break

mnist_train=MNIST(root="E:/pycharm_project/data/MNIST", train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
mnist_test=MNIST(root="E:/pycharm_project/data/MNIST",train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
# print(type(mnist_train))
# print(np.shape(mnist_train[0][0]))
# plt.imshow(mnist_train[0][0][0])
# plt.show()                                  #把图片展示出来
# print(mnist_train[0][1])                   #输出该图片的标签   [x][0]存放的是该图片，[x][1]存放的是该图片的标签

batchsize = 64
shuffle = True
epo = 10
learning_rate = 0.01
momentum = 0.5

# 导入数据加载器
train_loader = DataLoader(mnist_train, batch_size=batchsize, shuffle=shuffle)
test_loader = DataLoader(mnist_test, batch_size=batchsize, shuffle=shuffle)

# 查看数据具体情况
# for i, j in train_loader:
#     print(i)
#     print(j)
#     break

class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()    #继承torch.nn.Module的属性
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2,stride=2)

        self.fc1 = torch.nn.Linear(256, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84,10)


    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)                     # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x=self.relu1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        x=x.view(-1,4*4*16)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）

model=LeNet().cuda()
device = torch.device("cuda:0")
model.to(device)


criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0)   #T_max越大，lr下降的越慢


def l1_Regularization(model,lamda):
    l1=0
    for name,para in model.state_dict().items():
        if 'weight' in name:
            l1+=torch.sum(abs(para))
    return lamda*l1

def l2_Regularization(model,lamda):
    l2=0
    for name,para in model.state_dict().items():
        if 'weight' in name:
            l2+=torch.sum(pow(para,2))
    return  lamda*l2




def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        inputs=inputs.to(device)
        target=target.to(device)


        # forward + backward + update
        outputs = model(inputs)  #输出会以[0.11,0.15,0.90,0.87]这样的形式输出，每个数代表该图像属于这个标签的概率
        loss = criterion(outputs, target)+l2_Regularization(model,0.0001)
        optimizer.zero_grad()  # 梯度初始化为零，把loss关于权重weight的导数变成0
        loss.backward()   #反向传播得到每个参数的梯度值
        optimizer.step()  #通过梯度下降执行参数更新

        # 把运行中的loss累加起来,设定以100为一个轮回，以这100内的平均loss为loss值
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        values,predicted = torch.max(outputs.data, dim=1)    #求outputs.data的行或者列最大值，dim=1是行最大值，dim=0是列最大值，values收到每行最大值，predicted会收到最大值的索引，即求最大概率的标签
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 100 == 99:  # 设定每100次输出准确率和损失
            print('[{}, {}]: loss: {} , acc: {}%'.format(epoch + 1, batch_idx + 1, running_loss / 100, 100 * running_correct / running_total))

            #进行清零
            running_loss = 0.0
            running_total = 0
            running_correct = 0   #如果不清零则会进行累加

def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            outputs = model(images)
            values, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[{} / {}]: Accuracy on test set: {}% ' .format(epoch+1, epo, 100 * acc))
    return acc

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    acc_list_test = []
    for epoch in range(epo):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
