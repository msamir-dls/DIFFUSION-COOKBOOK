import torch
import torch.nn as nn
from src.models.unet import TimeEmbedding, ResidualBlock, Swish

class LatentUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Stable Diffusion latents usually have 4 channels
        c_in = config['vae']['latent_dim'] 
        base_ch = config['model']['base_channels']
        ch_mult = config['model']['channel_mult']
        num_res = config['model']['num_res_blocks']
        
        self.init_conv = nn.Conv2d(c_in, base_ch, 3, padding=1)
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList()
        curr_ch = base_ch
        feat_chs = [base_ch]
        
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(num_res):
                self.downs.append(ResidualBlock(curr_ch, out_ch, time_dim))
                curr_ch = out_ch
                feat_chs.append(curr_ch)
            if i != len(ch_mult) - 1:
                self.downs.append(nn.Conv2d(curr_ch, curr_ch, 3, stride=2, padding=1))
                feat_chs.append(curr_ch)

        self.mid = ResidualBlock(curr_ch, curr_ch, time_dim)
        
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            for _ in range(num_res + 1):
                skip_ch = feat_chs.pop()
                self.ups.append(ResidualBlock(curr_ch + skip_ch, out_ch, time_dim))
                curr_ch = out_ch
            if i != 0:
                self.ups.append(nn.ConvTranspose2d(curr_ch, curr_ch, 4, stride=2, padding=1))

        self.final = nn.Sequential(
            nn.GroupNorm(8, curr_ch),
            Swish(),
            nn.Conv2d(curr_ch, c_in, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        skips = [x]
        for layer in self.downs:
            x = layer(x, t_emb) if isinstance(layer, ResidualBlock) else layer(x)
            skips.append(x)
        
        x = self.mid(x, t_emb)
        
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return self.final(x)