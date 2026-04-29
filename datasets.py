import torch
from torchvision import datasets, transforms

def get_mnist(batch_size):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])

    dataset_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=6, persistent_workers=True)
    
    return train_loader, test_loader

def get_cifar10(batch_size):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    dataset_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataset_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    
    return train_loader, test_loader