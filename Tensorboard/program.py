import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def setup():
    # Transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Datasets
    trainset = torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # Classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneaker', 'Bag', 'Ankle Boot')

    return classes, trainloader, testloader


def mpl_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=2)
    img = img / 2 + 0.5  # Denormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

def main():
    net = Net()
    classes, trainloader, testloader = setup()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    #mpl_imshow(img_grid, one_channel=True)

    writer.add_image('four_fashion_mnist_images', img_grid)
    writer.add_graph(net, images)
    writer.close()

if __name__ == '__main__':
    main()
