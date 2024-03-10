import torch.nn as nn

class default_cnn(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 256, 5, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256*15*15, 256)
        self.fc2 = nn.Linear(256, out_size)
        self.flat = nn.Flatten()
        
    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.flat(X)
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        return X