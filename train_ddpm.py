import torch
import torch.nn.functional as F
import mlflow
from tqdm import tqdm
from src.utils import load_config, setup_mlflow, time_execution, save_checkpoint, get_device
from src.dataset import get_dataloader
from src.models.unet import UNet
from src.schedulers.gaussian import GaussianDiffusion

def main():
    config = load_config("configs/ddpm_train.yaml")
    device = get_device()
    dataloader = get_dataloader(config)
    
    model = UNet(config).to(device)
    diffusion = GaussianDiffusion(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    with mlflow.start_run(run_name="ddpm-pixel"):
        for epoch in range(1, config['train']['epochs'] + 1):
            model.train()
            pbar = tqdm(dataloader)
            total_loss = 0
            for x, _ in pbar:
                x = x.to(device)
                t = torch.randint(0, 1000, (x.size(0),), device=device).long()
                noise = torch.randn_like(x)
                x_noisy = diffusion.q_sample(x, t, noise)
                pred_noise = model(x_noisy, t)
                loss = F.mse_loss(pred_noise, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_description(f"DDPM Epoch {epoch} Loss: {loss.item():.4f}")
                
            mlflow.log_metric("loss", total_loss / len(dataloader), step=epoch)
            
        save_checkpoint(model, optimizer, epoch, "checkpoints/ddpm_mnist_final.pth")

if __name__ == "__main__":
    main()
