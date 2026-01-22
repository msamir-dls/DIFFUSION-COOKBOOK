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
def i2i_ddpm(model, diffusion, x_start, strength=0.5):
    """Pixel-space DDPM I2I translation."""
    device = x_start.device
    t_start = int(diffusion.timesteps * strength)
    t_idx = torch.full((x_start.shape[0],), t_start - 1, device=device).long()
    
    # Add noise to source image
    noisy_x = diffusion.q_sample(x_start, t_idx)
    
    # Denoise step-by-step
    img = noisy_x
    for i in reversed(range(0, t_start)):
        t = torch.full((x_start.shape[0],), i, device=device).long()
        img = diffusion.p_sample(model, img, t, i)
    return img

@time_execution
def i2i_ddim(model, ddim_solver, x_start, strength=0.5):
    """Pixel-space DDIM I2I translation (Accelerated)."""
    device = x_start.device
    t_start = int(ddim_solver.timesteps * strength)
    
    # Add noise
    t_idx = torch.full((x_start.shape[0],), t_start - 1, device=device).long()
    alpha_bar = ddim_solver.alphas_cumprod[t_idx].view(-1, 1, 1, 1)
    noisy_x = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * torch.randn_like(x_start)
    
    # Filter DDIM timesteps to only those below our start point
    time_pairs = ddim_solver.get_sampling_timesteps()
    time_pairs = [(t, tn) for t, tn in time_pairs if t < t_start]
    
    img = noisy_x
    for t, t_next in time_pairs:
        t_b = torch.full((x_start.shape[0],), t, device=device).long()
        tn_b = torch.full((x_start.shape[0],), t_next, device=device).long()
        img = ddim_solver.step(model, img, t_b, tn_b)
    return img

@time_execution
def i2i_stable_diffusion(latent_model, vae, diffusion, x_start, strength=0.5):
    """Latent-space (Stable Diffusion) I2I translation."""
    device = x_start.device
    # 1. Encode to Latent
    mu, logvar = vae.encode(x_start)
    latents = vae.reparameterize(mu, logvar)
    
    # 2. Add noise in latent space
    t_start = int(diffusion.timesteps * strength)
    t_idx = torch.full((latents.shape[0],), t_start - 1, device=device).long()
    noisy_latents = diffusion.q_sample(latents, t_idx)
    
    # 3. Denoise latents
    img_z = noisy_latents
    for i in reversed(range(0, t_start)):
        t = torch.full((latents.shape[0],), i, device=device).long()
        img_z = diffusion.p_sample(latent_model, img_z, t, i)
        
    # 4. Decode back to Pixels
    return vae.decode(img_z)

def main():
    config_ddpm = load_config("configs/ddpm_train.yaml")
    config_sd = load_config("configs/stable_diffusion.yaml")
    config_infer = load_config("configs/ddim_inference.yaml")
    device = get_device()

    # Load Models
    pixel_unet = UNet(config_ddpm).to(device)
    latent_unet = LatentUNet(config_sd).to(device)
    vae = VAE(config_sd).to(device)
    
    # Load weights (assuming you have trained these)
    # pixel_unet.load_state_dict(torch.load("checkpoints/ddpm_mnist_final.pth")['model_state_dict'])
    # latent_unet.load_state_dict(torch.load("checkpoints/sd_latent_final.pth")['model_state_dict'])
    
    # Schedulers
    diffusion_pixel = GaussianDiffusion(config_ddpm).to(device)
    ddim_solver = DDIMSolver(config_infer, diffusion_pixel.sqrt_alphas_cumprod**2)
    diffusion_latent = GaussianDiffusion(config_sd).to(device)

    # Prepare Data (8 source images)
    dataloader = get_dataloader(config_ddpm)
    source_images, _ = next(iter(dataloader))
    source_images = source_images[:8].to(device)

    mlflow.set_experiment("I2I_Comparison_Dashboard")
    with mlflow.start_run(run_name="i2i_speed_accuracy_benchmark"):
        # Run Comparisons
        res_ddpm = i2i_ddpm(pixel_unet, diffusion_pixel, source_images, strength=0.6)
        res_ddim = i2i_ddim(pixel_unet, ddim_solver, source_images, strength=0.6)
        res_sd = i2i_stable_diffusion(latent_unet, vae, diffusion_latent, source_images, strength=0.6)

        # Log Metrics (Accuracy as MSE vs Source)
        mlflow.log_metric("mse_ddpm", F.mse_loss(res_ddpm, source_images).item())
        mlflow.log_metric("mse_ddim", F.mse_loss(res_ddim, source_images).item())
        mlflow.log_metric("mse_sd", F.mse_loss(res_sd, source_images).item())

        # Save Visual Grid
        all_results = torch.cat([source_images, res_ddpm, res_ddim, res_sd], dim=0)
        grid = make_grid(all_results, nrow=8, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Source | DDPM | DDIM | Stable Diffusion")
        plt.savefig("i2i_comparison.png")
        mlflow.log_artifact("i2i_comparison.png")

if __name__ == "__main__":
    main()