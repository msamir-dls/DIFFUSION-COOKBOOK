import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os

class MNISTDiffusionDataset(Dataset):
    """
    A wrapper for MNIST that handles resizing and normalization to [-1, 1].
    """
    def __init__(self, root, img_size=32, train=True):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # Scale from [0, 1] to [-1, 1]
        ])
        
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # We only need the image for unconditional/image-to-image diffusion
        # but we return the label just in case you want to extend to class-conditioning later
        img, label = self.dataset[idx]
        return img, label

class LatentMNISTDataset(Dataset):
    """
    Used for Stable Diffusion. Instead of raw images, this stores the 
    latents produced by the VAE to speed up U-Net training.
    """
    def __init__(self, vae, dataloader, device):
        self.latents = []
        self.labels = []
        
        print("Encoding images into latent space...")
        vae.eval()
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(device)
                # Encode images to latent space
                moments = vae.encode(imgs)
                # For a simple VAE, we take the mean or sample
                latent = vae.reparameterize(moments) 
                self.latents.append(latent.cpu())
                self.labels.append(labels)
        
        self.latents = torch.cat(self.latents, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]

def get_dataloader(config):
    """
    Factory function to create dataloaders based on YAML config.
    """
    dataset = MNISTDiffusionDataset(
        root=config['dataset']['root'],
        img_size=config['dataset']['img_size'],
        train=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return loader