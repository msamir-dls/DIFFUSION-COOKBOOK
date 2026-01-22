import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class GaussianDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.timesteps = config['diffusion']['timesteps']
        beta_start = config['diffusion']['beta_start']
        beta_end = config['diffusion']['beta_end']

        # 1. Define beta schedule
        betas = torch.linspace(beta_start, beta_end, self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 2. Register buffers for the forward process (q)
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # 3. Register buffers for the reverse process (p)
        # x_{t-1} = sqrt_recip_alphas * (x_t - (betas / sqrt_one_minus_alphas_cumprod) * noise)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # This is used for the variance in the reverse step
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def _extract(self, a, t, x_shape):
        """Helper to extract values for a specific batch of timesteps."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """Forward process: adds noise to the original image (q(x_t | x_0))."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """Reverse process: one step of DDPM sampling (p(x_{t-1} | x_t))."""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in DDPM paper: Predicted mean of x_{t-1}
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            # Add noise for every step except the last one (Langevin dynamics)
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, shape):
        """Full DDPM Sampling Loop (1000 steps)."""
        device = next(model.parameters()).device
        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # Iterate backwards from T to 0
        for i in tqdm(reversed(range(0, self.timesteps)), desc='DDPM Sampling', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
        
        return img