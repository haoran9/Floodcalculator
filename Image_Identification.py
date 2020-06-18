import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler


def load_split_train_test(datadir, valid_size = .2, batch_size=64):
    train_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )
    train_data = datasets.ImageFolder(
        datadir,
        transform=train_transforms
    )
    test_data = datasets.ImageFolder(
        datadir,
        transform=test_transforms
    )
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=batch_size
    )
    testloader = torch.utils.data.DataLoader(
        test_data,
        sampler=test_sampler,
        batch_size=batch_size
    )
    return trainloader, testloader


def train(data_dir, model_name='model', vis=False):

    trainloader, testloader = load_split_train_test(data_dir, .2)
    print(trainloader.dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    print(model)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)

    epochs = 100
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals =top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))
                print("Epoch {} | Train loss {}.. ".format(epoch, running_loss))
                print("         | Test  loss {}.. ".format(test_loss))
                print("         | Accuracy   {}.. ".format(accuracy))

                running_loss = 0
                model.train()
    torch.save(model, '{}.pth'.format(model_name))

    if vis:
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()


if __name__ == "__main__":
    import sys
    train(data_dir=sys.argv[1])






