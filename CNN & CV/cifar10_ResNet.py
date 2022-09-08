import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 超参数定义
EPOCH = 10
batch_size = 64
lr = 0.001

# 数据加载
train_data = datasets.CIFAR10(root='/nas/cifar10/', train=True,
                              transform=transforms.ToTensor(),
                              download=True)

test_data = datasets.CIFAR10(root='/nas/cifar10/', train=False,
                             transform=transforms.ToTensor(),
                             download=True)

# 输出图像
temp = train_data[1][0].numpy()
# ic(temp.shape)
temp = temp.transpose(1, 2, 0)
# ic(temp.shape)
# plt.imshow(temp)
# plt.show()

# 使用DataLoader进行数据分批
train_load = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_load = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# 使用ResNet50
model = torchvision.models.resnet50(pretrained=False)
model = torchvision.models.resnet50(weights=None)


# 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 设置GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 模型和输入数据都需要to(device)
model = model.to(device)

# 训练过程（需要GPU高算力）
for epoch in range(EPOCH):
    for i, data in enumerate(train_load):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 前向传播
        outputs = model(inputs)
        # 计算损失函数
        loss = criterion(outputs, labels)
        # 清空上一轮梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
    print('epoch{} loss:{:.4f}'.format(epoch+1, loss.item()))

# 保存模型参数
torch.save(model, 'cifar10.ResNet.pt')
print('cifar10_ResNet.pt is saved')

# 模型加载
model = torch.load('cifar10_ResNet.pt')
# 测试
model.eval()
correct, total = 0, 0
for data in test_load:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    # 前向传播
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted==labels).sum().item()

# 输出测试集准确率
print('测试图像10000张的准确率：{:.4f}%'.format(100*correct/total))