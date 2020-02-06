import torch.nn as nn
import torch.nn.functional as F

'''
DropoutNet – add dropout layers to the BaseNet. The dropout should be placed on the output of the hidden
layers.
'''
class Net(nn.Module):
    def __init__(self,image_size):
        super(Net, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(100, 50)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(self.drop1(x)))
        x = F.relu(self.fc2(self.drop2(x)))
        return F.log_softmax(x, dim=1)
