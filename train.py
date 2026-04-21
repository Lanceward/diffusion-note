import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

dataset_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataset_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset_train)
test_loader = torch.utils.data.DataLoader(dataset_test)
