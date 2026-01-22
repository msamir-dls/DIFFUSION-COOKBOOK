import torch
import torch.nn.functional as F
import mlflow
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src.utils import load_config, time_execution, get_device
from src.models.unet import UNet
from src.models.vae import VAE
from src.models.latent_unet import LatentUNet
from src.schedulers.gaussian import GaussianDiffusion
from src.schedulers.ddim_solver import DDIMSolver
from src.dataset import get_dataloader

@time_execution
def i2i_ddpm(model, diffusion, x, strength=0.5):
    t_start = int(1000 * strength)
    t = torch.full((x.size(0),), t_start, device=x.device).long()
    noise = torch.randn_like(x)
    x_noisy = diffusion.q_sample(x, t, noise)
    
    img = x_noisy
    for i in reversed(range(t_start)):
        t_batch = torch.full((x.size(0),), i, device=x.device).long()
        img = diffusion.p_sample(model, img, t_batch, i)
    return img

@time_execution
def i2i_ddim(model, ddim_solver, x, strength=0.5):
    t_start = int(1000 * strength)
    t = torch.full((x.size(0),), t_start, device=x.device).long()
    
    # Add noise (forward process)
    # We cheat slightly by using the standard alpha schedules for q_sample 
    # but manually calculating here for consistency with DDIM math
    alpha_bar = ddim_solver.alphas_cumprod[t].view(-1, 1, 1, 1).to(x.device)
    noise = torch.randn_like(x)
    x_noisy = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
    
    # Filter steps to only run those < t_start
    time_pairs = ddim_solver.get_sampling_timesteps()
    relevant_steps = [(t_curr, t_prev) for t_curr, t_prev in time_pairs if t_curr < t_start]
    
    img = x_noisy
    for t_curr, t_prev in relevant_steps:
        t_c = torch.full((x.size(0),), t_curr, device=x.device).long()
        t_p = torch.full((x.size(0),), t_prev, device=x.device).long()
        img = ddim_solver.step(model, img, t_c, t_p)
    return img

@time_execution
def i2i_sd(model, vae, diffusion, x, strength=0.5):
    # Encode
    mu, logvar = vae.encode(x)
    z = vae.reparameterize(mu, logvar)
    
    # Noise Latents
    t_start = int(1000 * strength)
    t = torch.full((z.size(0),), t_start, device=x.device).long()
    noise = torch.randn_like(z)
    z_noisy = diffusion.q_sample(z, t, noise)
    
    # Denoise
    img_z = z_noisy
    for i in reversed(range(t_start)):
        t_batch = torch.full((z.size(0),), i, device=x.device).long()
        img_z = diffusion.p_sample(model, img_z, t_batch, i)
        
    return vae.decode(img_z)

def main():
    device = get_device()
    cfg_ddpm = load_config("configs/ddpm_train.yaml")
    cfg_sd = load_config("configs/stable_diffusion.yaml")
    cfg_infer = load_config("configs/ddim_inference.yaml")
    
    # Load Models
    unet = UNet(cfg_ddpm).to(device)
    if hasattr(torch, 'load'): 
        unet.load_state_dict(torch.load("checkpoints/ddpm_mnist_final.pth")['model_state_dict'])
    
    vae = VAE(cfg_sd).to(device)
    vae.load_state_dict(torch.load("checkpoints/sd_vae_mnist.pth"))
    
    latent_unet = LatentUNet(cfg_sd).to(device)
    latent_unet.load_state_dict(torch.load("checkpoints/sd_latent_final.pth")['model_state_dict'])
    
    # Schedulers
    diff_pixel = GaussianDiffusion(cfg_ddpm).to(device)
    ddim_solver = DDIMSolver(cfg_infer, diff_pixel.alphas_cumprod) # Pass alphas to DDIM
    diff_latent = GaussianDiffusion(cfg_sd).to(device)
    
    # Data
    x_real, _ = next(iter(get_dataloader(cfg_ddpm)))
    x_real = x_real[:8].to(device)
    
    # Run
    with mlflow.start_run(run_name="benchmark"):
        print("[*] Running DDPM (Standard)...")
        res_ddpm = i2i_ddpm(unet, diff_pixel, x_real, strength=0.6)
        
        print("[*] Running DDIM (Accelerated)...")
        res_ddim = i2i_ddim(unet, ddim_solver, x_real, strength=0.6)
        
        print("[*] Running Stable Diffusion (Latent)...")
        res_sd = i2i_sd(latent_unet, vae, diff_latent, x_real, strength=0.6)
        
        # Grid
        final = torch.cat([x_real, res_ddpm, res_ddim, res_sd], dim=0)
        grid = make_grid(final, nrow=8, normalize=True)
        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Source | DDPM | DDIM | Stable Diffusion")
        plt.axis('off')
        plt.savefig("i2i_comparison.png")
        mlflow.log_artifact("i2i_comparison.png")
        print("Comparison saved to i2i_comparison.png")

if __name__ == "__main__":
    main()
