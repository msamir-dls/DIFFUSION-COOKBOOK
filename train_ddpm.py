import torch
import torch.nn.functional as F
import mlflow
from tqdm import tqdm

from src.utils import load_config, setup_mlflow, time_execution, save_checkpoint, get_device
from src.dataset import get_dataloader
from src.models.unet import UNet
from src.schedulers.gaussian import GaussianDiffusion

@time_execution
def train_one_epoch(model, dataloader, diffusion, optimizer, device, config, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader)
    
    for x, _ in pbar:
        x = x.to(device)
        t = torch.randint(0, config['diffusion']['timesteps'], (x.shape[0],), device=device).long()
        
        noise = torch.randn_like(x)
        x_noisy = diffusion.q_sample(x_start=x, t=t, noise=noise)
        
        optimizer.zero_grad()
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def main():
    config = load_config("configs/ddpm_train.yaml")
    device = get_device()
    
    dataloader = get_dataloader(config)

    with mlflow.start_run(run_name=config['run_name']):
        # setup_mlflow(config)
        
        model = UNet(config).to(device)
        diffusion = GaussianDiffusion(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
        
        for epoch in range(1, config['train']['epochs'] + 1):
            avg_loss = train_one_epoch(model, dataloader, diffusion, optimizer, device, config, epoch)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            
            if epoch % 5 == 0:
                save_checkpoint(model, optimizer, epoch, f"checkpoints/ddpm_mnist_epoch_{epoch}.pth")

        # ---  Save with the exact name compare_inference.py expects ---
        save_checkpoint(model, optimizer, config['train']['epochs'], "checkpoints/ddpm_mnist_final.pth")

if __name__ == "__main__":
    main()
