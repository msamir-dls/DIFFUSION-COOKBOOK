import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['vae']['latent_dim'] 
        in_channels = config['dataset'].get('channels', 1)

        # Encoder: 32x32 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),          # 8x8
            nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),          # 8x8
            nn.BatchNorm2d(64), nn.LeakyReLU()
        )
        
        # Spatial Bottleneck (No Linear Layers!)
        self.conv_mu = nn.Conv2d(64, self.latent_dim, 3, padding=1)
        self.conv_logvar = nn.Conv2d(64, self.latent_dim, 3, padding=1)

        # Decoder: 8x8 -> 32x32
        self.decoder_input = nn.Conv2d(self.latent_dim, 64, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 16x16
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1), # 32x32
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.conv_mu(h), self.conv_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
