import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as ts
import torch

class Data(Dataset):
    def __init__(self, x, y):
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        normalize = ts.Normalize(mean=MEAN, std=STD)
        self.x = normalize(torch.Tensor(x).permute(0,3,1,2))
        self.y = torch.Tensor(y).to(dtype=int).squeeze(1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),    
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),   
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                
            nn.Conv2d(32, 64, 3, padding=1),   
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),   
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                
            nn.Conv2d(64, 128, 3, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 256, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x