import torch.nn as nn
import torch.nn.functional as F

'''
BaseNet - Neural Network with two hidden layers, the ﬁrst layer is of size 100 and the second layer
is of size 50, both followed by ReLU activation function.
'''
class Net(nn.Module):
    def __init__(self,image_size):
        super(Net, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
