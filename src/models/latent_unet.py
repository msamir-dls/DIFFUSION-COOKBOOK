import torch
import torch.nn as nn
from .unet import DownSample, UpSample, ResBlock, TimeEmbedding

class LatentUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # --- CRITICAL: Use Latent Dim (4), not Image Channels (1) ---
        in_channels = config['vae']['latent_dim']
        
        self.base_channels = config['model']['base_channels']
        self.timesteps = config['diffusion']['timesteps']
        
        # Time Embedding
        self.time_embed = TimeEmbedding(self.base_channels)
        
        # Initial Conv
        self.init_conv = nn.Conv2d(in_channels, self.base_channels, kernel_size=3, padding=1)
        
        # Downsampling
        self.downs = nn.ModuleList()
        ch = self.base_channels
        channel_mults = config['model']['channel_mult'] # e.g. [1, 2]
        
        current_mult = 1
        for i, mult in enumerate(channel_mults):
            out_ch = self.base_channels * mult
            for _ in range(config['model']['num_res_blocks']):
                self.downs.append(ResBlock(ch, out_ch, self.base_channels))
                ch = out_ch
            
            if i != len(channel_mults) - 1:
                self.downs.append(DownSample(ch))
                
        # Middle
        self.mid_block1 = ResBlock(ch, ch, self.base_channels)
        self.mid_attn = nn.Identity() # Simplification for MNIST
        self.mid_block2 = ResBlock(ch, ch, self.base_channels)
        
        # Upsampling
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = self.base_channels * mult
            for _ in range(config['model']['num_res_blocks'] + 1):
                self.ups.append(ResBlock(ch + out_ch, out_ch, self.base_channels)) # concat skip
                ch = out_ch
            
            if i != 0:
                self.ups.append(UpSample(ch))
                
        # Final
        self.final_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # Time
        t_emb = self.time_embed(t)
        
        # Initial
        x = self.init_conv(x)
        
        # Down
        skips = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, DownSample):
                x = layer(x)
                skips.append(x)
            else:
                x = layer(x)
        
        # Mid
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)
        
        return self.final_conv(x)
