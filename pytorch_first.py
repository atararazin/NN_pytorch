import torch
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import BaseNet
import BatchNormNet
import DropoutNet
import ReluNet
import SigmoidNet

def get_data_sets(transforms):
    transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            './data', train=True, download=True,transform=transforms),batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            './data', train=False, transform=transforms), batch_size=64, shuffle=True)
    return train_loader, test_loader

def train(epoch, model):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_data):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_data:        
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(test_data.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_data.dataset),
    100. * correct / len(test_data.dataset)))


train_data, test_data = get_data_sets(transforms)
nets = [BaseNet, DropoutNet, BatchNormNet, ReluNet ,SigmoidNet]
lr = 0.001

for net in nets:
    print(net.__name__)
    model = net.Net(image_size=28*28)
    optimizer = optim.Adam(model.parameters(), lr)
    for epoch in range(1, 10 + 1):
        train(epoch, model)
        test()
