import torch
import numpy as np
from torchvision import datasets, transforms
from diffusion_model import SimpleUNetModel

def get_mnist():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])

    dataset_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_train)
    test_loader = torch.utils.data.DataLoader(dataset_test)
    
    return train_loader, test_loader

if __name__=='__main__':
    train_loader, test_loader = get_mnist()
    T = 1000
    beta_1 = 1e-4
    beta_T = 0.02
    model = SimpleUNetModel(sample_size=28, in_channels=1, out_channels=1, dropout=0.1)
    
    # define the scheduler parameters
    # here for schedules, parameter_t is stored at index t-1
    beta = np.zeros(T)
    alpha = np.zeros(T)
    alpha_hat = np.zeros(T)
    for t in range(T):
        # parameters ranges from 1 to T
        beta[t] = (beta_T-beta_1)/T*t+beta_1
        alpha[t] = 1-beta[t]
        if t == 0:
            alpha_hat[t] = alpha[t]
        else:
            alpha_hat[t] = alpha_hat[t-1]*alpha[t]
        

    
