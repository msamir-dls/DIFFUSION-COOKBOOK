import torch
import torch.nn as nn
from .unet import ResBlock, TimeEmbedding

class LatentUNet(nn.Module):
    """Simplified U-Net for 4-channel latents"""
    def __init__(self, config):
        super().__init__()
        in_channels = config['vae']['latent_dim'] # 4
        base_channels = config['model']['base_channels']
        
        self.time_embed = TimeEmbedding(base_channels * 4)
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Simple U-Net structure for 8x8 input (no downsampling needed really, but we do one pass)
        self.down1 = ResBlock(base_channels, base_channels * 2, base_channels * 4)
        self.mid = ResBlock(base_channels * 2, base_channels * 2, base_channels * 4)
        self.up1 = ResBlock(base_channels * 4, base_channels, base_channels * 4) # 4*ch because concat
        
        self.final = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        t = self.time_embed(t)
        x = self.init_conv(x)      # [B, 64, 8, 8]
        
        h1 = self.down1(x, t)      # [B, 128, 8, 8]
        h_mid = self.mid(h1, t)    # [B, 128, 8, 8]
        
        # Concat skip connection
        h_up = torch.cat([h_mid, h1], dim=1) 
        h_up = self.up1(h_up, t)   # [B, 64, 8, 8]
        
        return self.final(h_up)
