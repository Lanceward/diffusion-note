import torch
import math
import matplotlib.pyplot as plt

def get_diffusion_scheduler_linear(T, beta_1, beta_T):
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
    
    return beta, alpha, alpha_hat

def get_diffusion_scheduler_cosine(T, s=0.008):
    def f(t):
        return math.cos((t/T+s)/(1+s) * math.pi/2)**2
    beta = torch.zeros(T+1)
    alpha = torch.ones(T+1)
    alpha_hat = torch.ones(T+1)
    for t in range(T+1):
        alpha_hat[t] = f(t)/f(0)
        if t > 0:
            alpha[t] = f(t)/f(t-1)#alpha_hat[t]/alpha_hat[t-1]
            beta[t] = min(1-alpha[t], 0.999)
    
    return beta, alpha, alpha_hat

if __name__=='__main__':
    beta, alpha, alpha_hat = get_diffusion_scheduler_cosine(T=1000)
    
    
    print(beta)
    print(alpha)
    print(alpha_hat)

    plt.plot(range(1001), beta)
    plt.plot(range(1001), alpha)
    plt.plot(range(1001), alpha_hat)
    plt.show()