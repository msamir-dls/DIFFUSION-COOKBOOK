import torch
import torch.nn as nn

class GaussianDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.timesteps = config['diffusion']['timesteps']
        self.beta_start = config['diffusion']['beta_start']
        self.beta_end = config['diffusion']['beta_end']
        
        # Linear schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha_bar = self.sqrt_alphas_cumprod.to(x_start.device)[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[t]
        
        # Reshape for broadcasting
        # Handle both 4D (images) and 2D (flat latents) inputs genericly
        shape = [x_start.shape[0]] + [1]*(len(x_start.shape)-1)
        
        return (
            sqrt_alpha_bar.view(*shape) * x_start +
            sqrt_one_minus_alpha_bar.view(*shape) * noise
        )

    def p_sample(self, model, x, t, t_index):
        # Simplified sampling step (DDPM)
        betas_t = self.betas.to(x.device)[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x.device)[t]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas.to(x.device)[t])
        
        # Model prediction
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alpha_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t * (1. - self.alphas_cumprod.to(x.device)[t-1]) / (1. - self.alphas_cumprod.to(x.device)[t])
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def sample(self, model, shape):
        device = next(model.parameters()).device
        img = torch.randn(shape, device=device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            with torch.no_grad():
                img = self.p_sample(model, img, t, i)
        return img
