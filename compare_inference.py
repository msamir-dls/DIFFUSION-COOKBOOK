import torch
import mlflow
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.utils import load_config, time_execution, get_device
from src.models.unet import UNet
from src.models.vae import VAE
from src.models.latent_unet import LatentUNet
from src.schedulers.gaussian import GaussianDiffusion
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
    
    # Load Models
    unet = UNet(cfg_ddpm).to(device)
    unet.load_state_dict(torch.load("checkpoints/ddpm_mnist_final.pth")['model_state_dict'])
    
    vae = VAE(cfg_sd).to(device)
    vae.load_state_dict(torch.load("checkpoints/sd_vae_mnist.pth"))
    
    latent_unet = LatentUNet(cfg_sd).to(device)
    latent_unet.load_state_dict(torch.load("checkpoints/sd_latent_final.pth")['model_state_dict'])
    
    # Schedulers
    diff_pixel = GaussianDiffusion(cfg_ddpm).to(device)
    diff_latent = GaussianDiffusion(cfg_sd).to(device)
    
    # Data
    x_real, _ = next(iter(get_dataloader(cfg_ddpm)))
    x_real = x_real[:8].to(device)
    
    # Run
    with mlflow.start_run(run_name="benchmark"):
        res_ddpm = i2i_ddpm(unet, diff_pixel, x_real)
        res_sd = i2i_sd(latent_unet, vae, diff_latent, x_real)
        
        # Grid
        final = torch.cat([x_real, res_ddpm, res_sd], dim=0)
        grid = make_grid(final, nrow=8, normalize=True)
        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Source | DDPM | Stable Diffusion")
        plt.axis('off')
        plt.savefig("i2i_comparison.png")
        print("Comparison saved to i2i_comparison.png")

if __name__ == "__main__":
    main()
