import torch
import torch.nn.functional as F
import mlflow
from tqdm import tqdm

from src.utils import load_config, setup_mlflow, time_execution, save_checkpoint, get_device
from src.dataset import get_dataloader, LatentMNISTDataset
from src.models.vae import VAE
from src.models.latent_unet import LatentUNet
from src.schedulers.gaussian import GaussianDiffusion

@time_execution
def train_latent_epoch(model, dataloader, diffusion, optimizer, device, config, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader)
    
    for latents, _ in pbar:
        latents = latents.to(device)
        
        t = torch.randint(0, config['diffusion']['timesteps'], (latents.shape[0],), device=device).long()
        
        noise = torch.randn_like(latents)
        x_noisy = diffusion.q_sample(x_start=latents, t=t, noise=noise)
        
        optimizer.zero_grad()
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"SD Latent Epoch {epoch} | Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def main():
    config = load_config("configs/stable_diffusion.yaml")
    device = get_device()
    
    # 1. Initialize VAE 
    vae = VAE(config).to(device)
    # Note: In a real app, you MUST load pretrained weights here or train the VAE first!
    # vae.load_state_dict(torch.load(config['vae']['pretrained_path']))
    
    pixel_loader = get_dataloader(config)
    
    # 2. Pre-encode Latents (This will now use the new Spatial VAE)
    latent_ds = LatentMNISTDataset(vae, pixel_loader, device)
    latent_loader = torch.utils.data.DataLoader(latent_ds, batch_size=config['train']['batch_size'], shuffle=True)

    with mlflow.start_run(run_name=config['run_name']):
        model = LatentUNet(config).to(device)
        diffusion = GaussianDiffusion(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
        
        for epoch in range(1, config['train']['epochs'] + 1):
            avg_loss = train_latent_epoch(model, latent_loader, diffusion, optimizer, device, config, epoch)
            mlflow.log_metric("latent_loss", avg_loss, step=epoch)
            
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    # Sample spatial latents [8, 4, 8, 8]
                    latent_shape = (8, config['vae']['latent_dim'], 8, 8)
                    z_samples = diffusion.sample(model, latent_shape)
                    pixel_samples = vae.decode(z_samples)
                    print(f"[*] Sampled {pixel_samples.shape} images.")

        save_checkpoint(model, optimizer, config['train']['epochs'], "checkpoints/sd_latent_final.pth")

if __name__ == "__main__":
    main()
