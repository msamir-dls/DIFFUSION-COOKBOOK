import torch
import torch.nn.functional as F
import mlflow
from tqdm import tqdm
from src.utils import load_config, setup_mlflow, time_execution, save_checkpoint, get_device
from src.dataset import get_dataloader, LatentMNISTDataset
from src.models.vae import VAE
from src.models.latent_unet import LatentUNet
from src.schedulers.gaussian import GaussianDiffusion

def main():
    config = load_config("configs/stable_diffusion.yaml")
    device = get_device()
    
    # Init VAE (Randomly initialized for this cookbook demo)
    vae = VAE(config).to(device)
    
    # Pre-compute Latents
    pixel_loader = get_dataloader(config)
    latent_ds = LatentMNISTDataset(vae, pixel_loader, device)
    latent_loader = torch.utils.data.DataLoader(latent_ds, batch_size=256, shuffle=True)
    
    model = LatentUNet(config).to(device)
    diffusion = GaussianDiffusion(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    with mlflow.start_run(run_name="sd-latent"):
        for epoch in range(1, config['train']['epochs'] + 1):
            model.train()
            pbar = tqdm(latent_loader)
            total_loss = 0
            for z, _ in pbar:
                z = z.to(device)
                t = torch.randint(0, 1000, (z.size(0),), device=device).long()
                noise = torch.randn_like(z)
                z_noisy = diffusion.q_sample(z, t, noise)
                pred_noise = model(z_noisy, t)
                loss = F.mse_loss(pred_noise, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_description(f"SD Epoch {epoch} Loss: {loss.item():.4f}")
                
            mlflow.log_metric("latent_loss", total_loss / len(latent_loader), step=epoch)

        save_checkpoint(model, optimizer, epoch, "checkpoints/sd_latent_final.pth")
        torch.save(vae.state_dict(), "checkpoints/sd_vae_mnist.pth")
        print("Models saved.")

if __name__ == "__main__":
    main()
