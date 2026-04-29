import os
import sys
import time

import argparse
import numpy as np
import torch

import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR, CosineAnnealingLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from diffusion_model import SimpleUNet2DModelGrey, SimpleUNet2DModelRGB, OpenAIUNet2DModelRGB
from datasets import get_mnist, get_cifar10
from diffusion_scheduler import get_diffusion_scheduler_linear, get_diffusion_scheduler_cosine

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Diffusion Training')
    parser.add_argument('-device', default='mps', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('-dataset', default='mnist', type=str, help='choice of datasets, default to mnist')
    parser.add_argument('-diff-schedule', default='linear', type=str, help='how the forward process add noise, defaul to linear schedule')
    parser.add_argument('-T', default=1000, type=int, help='diffusion timestep')

    args = parser.parse_args()
    print(args)
    
    #dataset and model
    if args.dataset == "mnist":
        train_loader, test_loader = get_mnist(batch_size=args.b)    
        net = SimpleUNet2DModelGrey(dims=(28, 28), num_class=10).to(args.device)
    elif args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10(batch_size=args.b)    
        # net = SimpleUNet2DModelRGB(dims=(32, 32), num_class=10).to(args.device)
        net = OpenAIUNet2DModelRGB(dims=(32, 32), num_class=10).to(args.device)
    else:
        raise ValueError(f"{args.dataset} is currently not supported")
    # define the scheduler parameters
    # here for schedules, parameter_t is stored at index t
    T = args.T
    if args.diff_schedule == 'linear':
        beta, alpha, alpha_hat = get_diffusion_scheduler_linear(T=T, beta_1=1e-4/(T/1000), beta_T=0.02/(T/1000))
    elif args.diff_schedule == 'cosine':
        beta, alpha, alpha_hat = get_diffusion_scheduler_cosine(T=T)
    else:
        raise ValueError(f"diffusion schedule {args.diff_schedule} not currently supported.")
    
    # Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    # Scheduler
    scheduler = ConstantLR(optimizer, factor=1.0)#MultiStepLR(optimizer, milestones=[50, 75])

    # preparation around training
    start_epoch=0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    
    out_dir = os.path.join(args.out_dir, f'{type(net).__name__}_T{T}_b{args.b}_lr{args.lr}_{args.diff_schedule}')
    out_dir += '_'+args.dataset

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

    #some pre-processing        
    dev = torch.device(args.device)
    alpha_hat = alpha_hat.to(dev)
    sqrt_alpha_hat = alpha_hat.sqrt().view(-1, 1, 1, 1).to(dev)
    sqrt_one_minus_alpha_hat = (1.0 - alpha_hat).sqrt().view(-1, 1, 1, 1).to(dev)
    
    #training starts
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        loop_time = time.time()
        net.train()
        idx = 0
        train_samples = 0
        train_loss = 0.0
        loss = torch.tensor(0.0)
        for img, label in train_loader:
            print(f'\r{idx}/{len(train_loader)} b:{img.shape[0]} loss:{loss: .4f} batchtime:{time.time()-loop_time: .4f} lr:{(optimizer.param_groups[0]['lr'])}', end='')
            idx+=1
            loop_time = time.time()
            optimizer.zero_grad(set_to_none=True)

            img = img.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)
            this_batch = img.shape[0]
            # print(img.shape, label.shape, end='')
            
            # generate t and epsilon for each image in barch
            ts = torch.randint(1, T+1, size=(this_batch,), device=dev)
            eps = torch.randn_like(img)
            # print(ts.shape, eps.shape, end='')
            
            # add noise
            c0 = sqrt_alpha_hat[ts]#.view(-1, 1, 1, 1)
            c1 = sqrt_one_minus_alpha_hat[ts]#.view(-1, 1, 1, 1)
            img_noised = c0 * img + c1 * eps
            
            #forward            
            pred_noise = net(sample=img_noised, timestep=ts, class_labels=label).sample
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
            'diffusion_scheduler': args.diff_schedule,
            'dataset': args.dataset,
            'T': args.T
        }
        
        torch.save(checkpoint, os.path.join(out_dir, f'checkpoint_epoch_{epoch}.pth'))

        print(args)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, training_speed ={train_speed: .4f}')
        