import os
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
    pbar = tqdm(dataloader)
    total_loss = 0
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        # Sample random timesteps for each image in the batch
        t = torch.randint(0, config['diffusion']['timesteps'], (images.shape[0],), device=device).long()
        
        # Forward Diffusion: Add noise
        noise = torch.randn_like(images)
        x_noisy = diffusion.q_sample(x_start=images, t=t, noise=noise)
        
        # Predict the noise added
        optimizer.zero_grad()
        predicted_noise = model(x_noisy, t)
        
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        
        # Log step-level loss to MLflow
        if batch_idx % 10 == 0:
            mlflow.log_metric("train_loss_step", loss.item())

    return total_loss / len(dataloader)

def main():
    # 1. Load configuration
    config = load_config("configs/ddpm_train.yaml")
    device = get_device()
    
    # 2. Setup MLflow
    mlflow.set_tracking_uri(config['mlflow'].get('tracking_uri', 'http://localhost:5000'))
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=config['run_name']):
        setup_mlflow(config) # Logs all params from YAML
        
        # 3. Initialize Data, Model, and Diffusion
        dataloader = get_dataloader(config)
        model = UNet(config).to(device)
        diffusion = GaussianDiffusion(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
        
        print(f"[*] Starting DDPM Training on {device}...")
        
        # 4. Training Loop
        for epoch in range(1, config['train']['epochs'] + 1):
            avg_loss = train_one_epoch(model, dataloader, diffusion, optimizer, device, config, epoch)
            
            # Log epoch metrics
            mlflow.log_metric("avg_loss_epoch", avg_loss, step=epoch)
            
            # 5. Save Checkpoint
            if epoch % config['train']['save_every'] == 0:
                ckpt_path = f"checkpoints/ddpm_mnist_epoch_{epoch}.pth"
                save_checkpoint(model, optimizer, epoch, ckpt_path)
                
                # Sample images to log to MLflow for visual tracking
                model.eval()
                with torch.no_grad():
                    # (Batch, Channels, H, W)
                    sample_shape = (8, config['dataset']['channels'], 
                                    config['dataset']['img_size'], 
                                    config['dataset']['img_size'])
                    samples = diffusion.sample(model, sample_shape)
                    # Unnormalize [-1, 1] to [0, 1]
                    samples = (samples.clamp(-1, 1) + 1) / 2
                    
                    # You could use torchvision.utils.save_image here to log to MLflow
                    # mlflow.log_artifact(local_path_of_grid_image)

        # Save final model
        final_path = "checkpoints/ddpm_mnist_final.pth"
        save_checkpoint(model, optimizer, config['train']['epochs'], final_path)
        mlflow.log_artifact(final_path)

if __name__ == "__main__":
    main()