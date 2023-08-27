import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifierANN(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        
        
class MNISTClassifierCNN(torch.nn.Module):
    
    def __init__(self, in_channels, out_features):
        super().__init__()
        
        # First Layer
        # ImgIn shape = (?, 28, 28, 1)
        #       Conv -> (?, 28, 28, 32)
        #       Pool -> (?, 14, 14, 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), # 32 filters -> resulting size = (28+1*2-3)/1+1, (28+1*2-3)/1+1, 32 = (28,28,32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (32,28,28) -> (28-2)/2+1, (28-2)/2+1 = (14,14,32)
        )
        # Second Layer
        # ImgIn shape = (?, 14, 14, 32)
        #       Conv -> (?, 14, 14, 64)
        #       Pool -> (?,  7,  7, 64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, out_features)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x