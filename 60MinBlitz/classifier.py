import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main(PATH, classes, device):
    print("Creating transforms for image dataset")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print("Creating dataset object")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    print("Creating dataloader with a batch size of 4")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    print(f"Classes are {classes}")

    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    #
    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    print("Create network")
    net = Net()
    net.to(device)

    print("Define loss (cross entropy) and optimizer")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9)

    print("iterate over number of epochs and dataset")
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1, i + 1}] loss: {running_loss / 2000}")
                running_loss = 0.0

    print("finished training")

    print("save the model")

    torch.save(net.state_dict(), PATH)
    print()
    return testloader


def imshow(img):
    img = img / 2 + 0.5  # denormalize...
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def inference(testloader, PATH, classes):
    correct = 0
    total = 0

    net = Net()
    net.load_state_dict(torch.load(PATH))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            #print(f"output size: {outputs.size()}")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%")

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]} %")


if __name__ == '__main__':
    PATH = './cifar_net.pth'
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print()
    print(f"Device is: {device}")
    tloader = main(PATH, classes, device)
    print()
    print("Inference time...")
    inference(tloader, PATH, classes)


