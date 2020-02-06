import torch.nn as nn
import torch.nn.functional as F
'''
BatchNormNet - add Batch Normalization layers to BaseNet. The Batch Normalization should be placed 
before the activation functions.
'''
class Net(nn.Module):
    def __init__(self,image_size):
        super(Net, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.bn0 = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
