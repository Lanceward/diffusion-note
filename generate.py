import torch
from diffusion_model import SimpleUNet2DModelGrey
import matplotlib.pyplot as plt

DIGIT = 6

if __name__=="__main__":
    T = 1000
    beta_1 = 1e-4
    beta_T = 0.02
    model_path = "./logs/SimpleUNetModel_T1000_b128_lr0.0001_mnist/checkpoint_epoch_99.pth"
    DEV = "mps"
    model = SimpleUNet2DModelGrey(dims=(28, 28), num_class=10).to(DEV)
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=DEV)['net'])
    model.eval()
    model.to(DEV)
    # print(model)
    
    # define the scheduler parameters
    # here for schedules, parameter_t is stored at index t
    beta = torch.zeros(T+1).to(DEV)
    alpha = torch.zeros(T+1).to(DEV)
    alpha_hat = torch.zeros(T+1).to(DEV)
    for t in range(1, T+1):
        # parameters ranges from 1 to T
        beta[t] = (beta_T-beta_1)/(T-1)*(t-1)+beta_1
        alpha[t] = 1-beta[t]
        if t == 1:
            alpha_hat[t] = alpha[t]
        else:
            alpha_hat[t] = alpha_hat[t-1]*alpha[t]
    # torch.set_printoptions(precision=10)

    # print(beta)
    # print(alpha)
    # print(alpha_hat)

    x_t = torch.randn(1, 1, 28, 28, device=DEV)
    class_label = torch.tensor(DIGIT, device=DEV)

    plt.ion() # Turn interactive mode on
    fig, ax = plt.subplots()
    img_data = x_t.detach().cpu().squeeze(0, 1).numpy()
    im = ax.imshow(img_data)#, cmap='gray')
    for t in range(T, 0, -1):
        print(f'\r {t}/{T} mean: {x_t.mean()}, min|max: {x_t.min()}|{x_t.max()}', end='      ')
        if t > 1:
            z = torch.randn(1, 1, 28, 28, device=DEV)
        else:
            z = torch.zeros((1, 1, 28, 28), device=DEV)
        
        pred_noise = model.forward(sample=x_t, timestep=t, class_labels=class_label).sample
        x_t_1 = 1/torch.sqrt(alpha[t])*(x_t - (1-alpha[t])/torch.sqrt(1-alpha_hat[t])*pred_noise) + torch.sqrt(beta[t])*z
        x_t = x_t_1
        
        #plot progression
        new_data = x_t.detach().cpu().squeeze(0, 1).numpy()
        im.set_data(new_data)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    print()
    
    plt.ioff()
    plt.show()
    # # show the image
    # img_to_show = x_t.detach().cpu().squeeze(0, 1).numpy()
    # plt.imshow(img_to_show, cmap='gray')
    # plt.axis('off') # Optional: hides axes
    # plt.show()