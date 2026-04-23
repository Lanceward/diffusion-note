import argparse
import torch

from diffusion_model import SimpleUNet2DModelGrey, SimpleUNet2DModelRGB
import matplotlib.pyplot as plt
from diffusion_scheduler import get_diffusion_scheduler_linear, get_diffusion_scheduler_cosine

LIVE = False

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Diffusion Training')
    parser.add_argument('-device', default='mps', help='device')
    parser.add_argument('-classidx', default=0, type=int, help='class index')    
    parser.add_argument('-dataset', default='mnist', type=str, help='choice of datasets, default to mnist')
    parser.add_argument('-diff-schedule', default='linear', type=str, help='how the forward process add noise, defaul to linear schedule')

    args = parser.parse_args()
    print(args)

    DEV = args.device
    INDEX = args.classidx
    if args.dataset == "mnist":
        # model_path = "./logs/SimpleUNetModel_T1000_b128_lr0.0001_mnist/checkpoint_epoch_99.pth"
        model_path = "./logs/SimpleUNet2DModelGrey_T4000_b128_lr0.0001_cosine_mnist/checkpoint_epoch_51.pth"
        model = SimpleUNet2DModelGrey(dims=(28, 28), num_class=10).to(DEV)
        x_t = torch.randn(1, 1, 28, 28, device=DEV)
        if INDEX >= 0 and INDEX < 10:
            class_label = torch.tensor(INDEX, device=DEV)
        else:
            raise ValueError(f"{INDEX} is out of range for classes in {args.dataset}")
    elif args.dataset == "cifar10":
        # model_path = "./logs/SimpleUNet2DModelRGB_T1000_b128_lr0.0001_cifar10/checkpoint_epoch_99.pth"
        model_path = "./logs/SimpleUNet2DModelRGB_T1000_b128_lr0.0001_cosine_cifar10/checkpoint_epoch_99.pth"
        model = SimpleUNet2DModelRGB(dims=(32, 32), num_class=10).to(DEV)
        x_t = torch.randn(1, 3, 32, 32, device=DEV)
        class_label = torch.tensor(INDEX, device=DEV)
        if INDEX >= 0 and INDEX < 10:
            class_label = torch.tensor(INDEX, device=DEV)
        else:
            raise ValueError(f"{INDEX} is out of range for classes in {args.dataset}")
    else:
        raise ValueError(f"{args.dataset} is currently not supported")
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=DEV)['net'])
    model.eval()
    model.to(DEV)
    
    # define the scheduler parameters
    # here for schedules, parameter_t is stored at index t
    T = 4000
    if args.diff_schedule == 'linear':
        beta, alpha, alpha_hat = get_diffusion_scheduler_linear(T=T, beta_1=1e-4, beta_T=0.02)
    elif args.diff_schedule == 'cosine':
        beta, alpha, alpha_hat = get_diffusion_scheduler_cosine(T=T)
    else:
        raise ValueError(f"diffusion schedule {args.diff_schedule} not currently supported.")

    beta = beta.to(DEV)
    alpha = alpha.to(DEV)
    alpha_hat = alpha_hat.to(DEV)

    # print(beta)
    # print(alpha)
    # print(alpha_hat)

    if LIVE:
        plt.ion() # Turn interactive mode on
        fig, ax = plt.subplots()
        img_data = x_t.detach().cpu().squeeze(0, 1).numpy()
        im = ax.imshow(img_data)#, cmap='gray')
    with torch.no_grad():
        for t in range(T, 0, -1):
            print(f'\r {t}/{T} mean: {x_t.mean()}, min|max: {x_t.min()}|{x_t.max()}', end='      ')
            if t > 1:
                z = torch.randn_like(x_t, device=DEV)
            else:
                z = torch.zeros_like(x_t, device=DEV)
            
            pred_noise = model.forward(sample=x_t, timestep=t, class_labels=class_label).sample
            x_t = 1/torch.sqrt(alpha[t])*(x_t - (1-alpha[t])/torch.sqrt(1-alpha_hat[t])*pred_noise) + torch.sqrt(beta[t])*z
            # x_t = x_t_1
            
            if LIVE:
                # plot progression
                new_data = x_t.detach().cpu().squeeze(0, 1).numpy()
                im.set_data(new_data)
                fig.canvas.draw()
                fig.canvas.flush_events()
        print()

    if LIVE:
        plt.ioff()
        plt.show()
        
    # show the image
    img = (x_t+1.0)/2.0
    # img = (x_t - x_t.min())/(x_t.max()-x_t.min())
    img_to_show = img.detach().cpu().squeeze(0).numpy()
    if img_to_show.shape[0] == 1:
        plt.imshow(img_to_show.squeeze(0), cmap='gray')
    elif img_to_show.shape[0] == 3:
        plt.imshow(img_to_show.transpose((1, 2, 0)), vmin=x_t.min(), vmax=x_t.max())
    else:
        raise ValueError(f"Current image shape {img_to_show.shape} not supported")
    plt.axis('off') # Optional: hides axes
    plt.savefig('image.png')
