import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 32x32x1 -> 16x16x32 -> 8x8xlatent_dim
        ld = config['vae']['latent_dim']
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, ld * 2) # Output mean and log-var
        )
        
        self.decoder_input = nn.Linear(ld, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), # 32x32
            nn.Tanh() # Output in range [-1, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar