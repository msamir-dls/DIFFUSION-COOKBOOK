import torch
import numpy as np
from tqdm import tqdm

class DDIMSolver:
    def __init__(self, config, alpha_cumprod):
        """
        alpha_cumprod: should be passed from the GaussianDiffusion instance 
                      (diffusion.sqrt_alphas_cumprod ** 2)
        """
        self.alphas_cumprod = alpha_cumprod
        self.timesteps = config['diffusion']['timesteps']
        self.ddim_timesteps = config['sampling']['steps'] 
        self.eta = config['sampling'].get('eta', 0.0) 

    def get_sampling_timesteps(self):
        # Create a subsequence of timesteps (e.g., jumping by 20 if steps=50)
        times = np.linspace(0, self.timesteps - 1, self.ddim_timesteps).astype(int)
        times = times.tolist()
        # Create pairs of (t, t_prev)
        return list(reversed(list(zip(times[1:], times[:-1]))))

    @torch.no_grad()
    def step(self, model, x, t, t_next):
        # 1. Predict noise using the U-Net
        et = model(x, t)
        
        # 2. Extract alpha_bar for current and next step
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_next = self.alphas_cumprod[t_next].view(-1, 1, 1, 1) if t_next >= 0 else torch.ones_like(alpha_t)

        # 3. Predict "predicted x0" (Equation 12 in DDIM paper)
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * et) / torch.sqrt(alpha_t)
        
        # 4. Compute variance (sigma). If eta=0, sigma=0 (Deterministic)
        sigma_t = self.eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
        
        # 5. Compute "direction pointing to xt"
        dir_xt = torch.sqrt(1 - alpha_next - sigma_t**2) * et
        
        # 6. Combine
        x_next = torch.sqrt(alpha_next) * pred_x0 + dir_xt
        
        if self.eta > 0:
            noise = torch.randn_like(x)
            x_next += sigma_t * noise
            
        return x_next

    @torch.no_grad()
    def sample(self, model, shape):
        """Full DDIM Sampling Loop (Faster than DDPM)."""
        device = next(model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        time_pairs = self.get_sampling_timesteps() # [(999, 979), (979, 959)...]
        
        for t, t_next in tqdm(time_pairs, desc=f'DDIM Sampling ({self.ddim_timesteps} steps)'):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            t_next_batch = torch.full((b,), t_next, device=device, dtype=torch.long)
            img = self.step(model, img, t_batch, t_next_batch)
            
        return img