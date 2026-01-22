import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def get_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize((config['dataset']['img_size'], config['dataset']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(
        root=config['dataset']['root'], 
        train=True, 
        download=True, 
        transform=transform
    )
    
    return DataLoader(
        dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True,
        drop_last=True
    )

class LatentMNISTDataset(Dataset):
    def __init__(self, vae, dataloader, device):
        self.latents = []
        print("Encoding images into latent space...")
        vae.eval()
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(device)
                mu, logvar = vae.encode(imgs)
                # Unpack explicitly
                latent = vae.reparameterize(mu, logvar)
                self.latents.append(latent.cpu())
                
        self.latents = torch.cat(self.latents, dim=0)
        print(f"Latent Dataset Shape: {self.latents.shape}")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], 0 # Dummy label
