import torch.nn as nn

class default_cnn(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, in_size, 3, 1)
        self.conv2 = nn.Conv2d(in_size, in_size*2, 5, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_size*2*15*15, out_size) #TODO
        self.flat = nn.Flatten()
        
    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.flat(X)
        X = self.fc(X)
        return X