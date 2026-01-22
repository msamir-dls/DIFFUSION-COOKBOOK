import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config params
        self.latent_dim = config['vae']['latent_dim'] # Should be 4
        in_channels = config['dataset'].get('channels', 1)

        # --- ENCODER (Fully Convolutional) ---
        # Input: [B, 1, 32, 32]
        self.encoder = nn.Sequential(
            # Layer 1: 32x32 -> 16x16
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # Layer 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # Layer 3: 8x8 -> 8x8 (Process features)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        # Bottleneck: Map 64ch -> latent_dim (4ch) mean and logvar
        # We keep it spatial (8x8), so no Linear layers!
        self.conv_mu = nn.Conv2d(64, self.latent_dim, kernel_size=3, padding=1)
        self.conv_logvar = nn.Conv2d(64, self.latent_dim, kernel_size=3, padding=1)

        # --- DECODER (Mirror of Encoder) ---
        self.decoder_input = nn.Conv2d(self.latent_dim, 64, kernel_size=3, padding=1)
        
        self.decoder = nn.Sequential(
            # Layer 1: 8x8 -> 8x8
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # Layer 2: 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # Layer 3: 16x16 -> 32x32
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh() # Output -1 to 1
        )

    def encode(self, x):
        # x: [B, 1, 32, 32]
        h = self.encoder(x)
        # mu, logvar: [B, 4, 8, 8] -> Spatial Latents!
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: [B, 4, 8, 8]
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
