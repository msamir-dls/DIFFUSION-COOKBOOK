import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    def forward(self, t):
        # Sinusoidal embedding
        device = t.device
        half_dim = self.dim // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.mlp(emb)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        h += self.mlp(t)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config['dataset']['channels']
        base_channels = config['model']['base_channels']
        mults = config['model']['channel_mult']
        
        self.time_embed = TimeEmbedding(base_channels * 4)
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Down
        ch = base_channels
        dims = [ch]
        for mult in mults:
            out_ch = base_channels * mult
            for _ in range(config['model']['num_res_blocks']):
                self.downs.append(ResBlock(ch, out_ch, base_channels * 4))
                ch = out_ch
                dims.append(ch)
            if mult != mults[-1]:
                self.downs.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                dims.append(ch)
                
        # Mid
        self.mid1 = ResBlock(ch, ch, base_channels * 4)
        self.mid2 = ResBlock(ch, ch, base_channels * 4)
        
        # Up
        for mult in reversed(mults):
            out_ch = base_channels * mult
            for _ in range(config['model']['num_res_blocks'] + 1): # +1 for concat
                in_ch = dims.pop() + ch
                self.ups.append(ResBlock(in_ch, out_ch, base_channels * 4))
                ch = out_ch
            
            if mult != mults[0]:
                self.ups.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                
        self.final = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        t = self.time_embed(t)
        x = self.init_conv(x)
        skips = [x]
        
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
                skips.append(x)
            else:
                x = layer(x)
                skips.append(x)
                
        x = self.mid1(x, t)
        x = self.mid2(x, t)
        
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = layer(x, t)
            else:
                x = layer(x)
                
        return self.final(x)
