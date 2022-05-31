from models import Data, VGGNet
from torch.utils.data import DataLoader
import numpy as np
import torch

x_test = np.load('./data/x_test.npy')
y_test = np.load('./data/y_test.npy')

test_set = Data(x_test, y_test)
test_loader = DataLoader(test_set,batch_size=16, shuffle=False)

net = VGGNet(10).to('cuda')
net.load_state_dict(torch.load("./saved/model.pt"))
net.eval()

correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        images, labels = data
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = net(images)
        _, predicts = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum().item()

acc = correct / total
print(acc)

