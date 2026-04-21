import os
import sys
import time

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR, CosineAnnealingLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from diffusion_model import SimpleUNetModel
# from diffusers import UNet2DModel

def get_mnist(batch_size):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])

    dataset_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    
    return train_loader, test_loader

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Diffusion MNIST Training')
    parser.add_argument('-device', default='mps', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-lr', default=2e-5, type=float, help='learning rate')

    args = parser.parse_args()
    print(args)
    
    T = 1000
    beta_1 = 1e-4
    beta_T = 0.02
    train_loader, test_loader = get_mnist(batch_size=args.b)    
    net = SimpleUNetModel(
        sample_size=28, 
        in_channels=1, 
        out_channels=1, 
        dropout=0.1, 
        num_class_embeds=10, 
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels=(32, 64, 64),
    ).to(args.device)
    
    # define the scheduler parameters
    # here for schedules, parameter_t is stored at index t
    beta = torch.zeros(T+1)
    alpha = torch.zeros(T+1)
    alpha_hat = torch.zeros(T+1)
    for t in range(1, T+1):
        # parameters ranges from 1 to T
        beta[t] = (beta_T-beta_1)/(T-1)*(t-1)+beta_1
        alpha[t] = 1-beta[t]
        if t == 1:
            alpha_hat[t] = alpha[t]
        else:
            alpha_hat[t] = alpha_hat[t-1]*alpha[t]
    
    # print(beta)
    # print(alpha)
    # print(alpha_hat)
    # exit()
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # Scheduler
    # warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    # main_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=100)
    # main_scheduler = CosineAnnealingLR(optimizer, T_max=100)
    scheduler = MultiStepLR(optimizer, milestones=[25, 50, 75])
    # scheduler = SequentialLR(
    #     optimizer, 
    #     schedulers=[warmup_scheduler, main_scheduler], 
    #     milestones=[10]
    # )

    # preparation around training
    start_epoch=0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    
    out_dir = os.path.join(args.out_dir, f'{type(net).__name__}_T{T}_b{args.b}_lr{args.lr}')
    out_dir += '_mnist'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))
        
    #training starts
    dev = torch.device(args.device)
    alpha_hat = alpha_hat.to(args.device)
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        idx = 0
        train_samples = 0
        train_loss = 0.0
        loss = 0.0
        for img, label in train_loader:
            print("\r" + str(idx) + "/" + str(len(train_loader)) + ' ' + str(img.shape) + " loss " + str(loss.item()), end='')
            idx+=1
            img = img.to(args.device)
            label = label.to(args.device)
            this_batch = img.shape[0]
            # print(img.shape, label.shape, end='')
            
            # generate t and epsilon for each image in barch
            ts = torch.randint(1, high=T+1, size=(this_batch,)).to(args.device)
            eps = torch.normal(mean=torch.zeros(this_batch, 1, 28, 28), std=torch.ones(this_batch, 1, 28, 28)).to(args.device)
            # print(ts.shape, eps.shape, end='')
            
            # add noise
            ahats = alpha_hat[ts]
            img_noised = torch.sqrt(ahats).view(-1, 1, 1, 1) * img + torch.sqrt(1-ahats).view(-1, 1, 1, 1) * eps
            
            #forward            
            pred_noise = net.forward(sample=img_noised, timestep=ts, class_labels=label).sample
            # print(pred_noise.shape, end='')
            loss = F.mse_loss(pred_noise, eps)
            
            #backward
            loss.backward()
            optimizer.step()
            
            #statistics
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
        print()

        #epoch stats
        scheduler.step()
        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        
        writer.add_scalar('train_loss', train_loss, epoch)
        
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
        }
        
        torch.save(checkpoint, os.path.join(out_dir, f'checkpoint_epoch_{epoch}.pth'))

        print(args)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}')
        