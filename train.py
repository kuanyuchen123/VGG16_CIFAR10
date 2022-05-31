import torch
import torch.nn as nn
from models import Data, VGGNet
import numpy as np
from torch.utils.data import DataLoader
from torch import optim

x_train = np.load('./data/x_train.npy')
y_train = np.load('./data/y_train.npy')

train_set = Data(x_train, y_train)
train_loader = DataLoader(train_set,batch_size=16, shuffle=True)

net = VGGNet(10).to('cuda')
net.train()
lr = 1e-3
momentum = 0.9
num_epoch = 100

critierion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
print('Training with learning rate = %f, momentum = %f ' % (lr, momentum))

for t in range(num_epoch):
    running_loss = 0
    running_loss_sum_per_epoch = 0
    total_images = 0
    correct_images = 0

    if t == 25:
        optimizer = optim.SGD(net.parameters(), lr=lr/10, momentum=momentum)
        
    for i, data in enumerate(train_loader, 0):
        images, labels = data

        optimizer.zero_grad()
        images = images.to('cuda')
        labels = labels.to('cuda')

        outputs = net(images)
        loss = critierion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_images += labels.size(0)
        _, predicts = torch.max(outputs.data, 1)

        correct_images += (predicts == labels).sum().item()
        running_loss += loss.data.item()
        running_loss_sum_per_epoch = running_loss + running_loss_sum_per_epoch

        if i % 2000 == 1999:
            print('Epoch, batch [%d, %5d] loss: %.6f, Training accuracy: %.5f' %
                  (t + 1, i + 1, running_loss / 2000, 100 * correct_images / total_images))
            running_loss = 0
            total_images = 0
            correct_images = 0

    torch.save(net.state_dict(), "./saved/model{}.pt".format(t))

print('Finished training')
