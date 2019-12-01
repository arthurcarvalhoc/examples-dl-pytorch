import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from Net import Net
from torch.autograd import Variable

input_size = 784
hidden_size = 400
out_size = 10
epochs = 10
batch_size = 100
lr = 0.1

train_dataset = datasets.MNIST(root='./data', train=True, 
                               transform=transforms.ToTensor(), 
                               download=True)
test_dataset = datasets.MNIST(root='./data', train=False, 
                              transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

print(train_loader)

net = Net(input_size, hidden_size, out_size)
criterion = nn.CrossEntropyLoss()
optmizer = torch.optim.Adam(net.parameters(), lr=lr)

correct_train = 0
total_train = 0

for i, (images, lables) in enumerate(train_loader):
    # print("{} - {} - {}".format(i, images.size(), lables.size()))
    images = images.view(-1, 1*28*28)
    # print(images.size())

print("TRAINING ...")
for epoch in range(epochs):
    print(".")
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        lables = Variable(lables)

        optmizer.zero_grad()
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total_train += labels.size(0)

        correct_train += (predicted + labels).sum()

        loss = criterion(outputs, labels)
        loss.backward()
        optmizer.step()

        if(i+1) % 100 == 0:
            print("Epoch {}/{}, Iteration {}/{}, TrainLoss {}, TrainAccuracy {}%".format(epoch, epochs, i+1, len(train_dataset)//batch_size, 0, correct_train/total_train))
            # print("Epoch {}/{}, Iteration {}/{}, TrainLoss {}, TrainAccuracy {}%".format(epoch, epochs, i+1, len(train_dataset)//batch_size, loss.data[0], (100*correct_train/total_train)))
    print("DONE TRAINING!")
