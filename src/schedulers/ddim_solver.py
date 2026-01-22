import torch
import numpy as np

class DDIMSolver:
    def __init__(self, config, alphas_cumprod):
        self.timesteps = 1000 # Assumed total training steps
        self.ddim_timesteps = config['sampling']['steps']
        self.eta = config['sampling']['eta']
        self.alphas_cumprod = alphas_cumprod

        # Calculate stepping (e.g., 0, 20, 40...)
        c = self.timesteps // self.ddim_timesteps
        self.ddim_steps = np.asarray(list(range(0, self.timesteps, c))) + 1
        
    def get_sampling_timesteps(self):
        # Return list of (t, t_prev) pairs reversed
        steps = np.flip(self.ddim_steps)
        time_pairs = []
        for i, step in enumerate(steps[:-1]):
            time_pairs.append((step, steps[i+1]))
        return time_pairs

    def step(self, model, x, t, t_prev):
        # Implementation of DDIM non-markovian step
        device = x.device
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
        alpha_bar_t_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1).to(device)
        sigma = self.eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
        
        # Predict noise
        epsilon = model(x, t)
        
        # Predict x0
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * epsilon) / torch.sqrt(alpha_bar_t)
        
        # Direction to xt
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma**2) * epsilon
        
        # Noise
        noise = sigma * torch.randn_like(x)
        
        return torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + noise
