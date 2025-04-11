import torch
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self, im_chan, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.disc = nn.Sequential(
        # Input: (input_dim, 28, 28) -> Output: (hidden_dim, 14, 14)
            nn.Conv2d(in_channels=im_chan, out_channels=hidden_dim,
                    kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        # Input: (hidden_dim, 14, 14) -> Output: (hidden_dim * 2, 7, 7)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2,
                    kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

        # Input: (hidden_dim * 2, 7, 7) -> Output: (hidden_dim * 4, 3, 3)
            nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4,
                    kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

        # Input: (hidden_dim * 4, 3, 3) -> Output: (z_dim, 1, 1)
            nn.Conv2d(in_channels=hidden_dim * 4, out_channels=z_dim,
                    kernel_size=3, stride=1, padding=0, bias=False)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.disc(image)
        return x

class Decoder(nn.Module):
    def __init__(self, z_dim, im_chan, hidden_dim):
        super(Decoder, self).__init__()

        self.gen = nn.Sequential(
        # Input: (z_dim, 1, 1) -> Output: (hidden_dim * 4, 3, 3)
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=hidden_dim * 4,
                                kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),

        # Input: (hidden_dim * 4, 3, 3) -> Output: (hidden_dim * 2, 7, 7)
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2,
                            kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),

        # Input: (hidden_dim * 2, 7, 7) -> Output: (hidden_dim, 14, 14)
            nn.ConvTranspose2d(in_channels=hidden_dim * 2, out_channels=hidden_dim,
                            kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

        # Input: (hidden_dim, 14, 14) -> Output: (im_chan, 28, 28)
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=im_chan,
                            kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, latent_vec: torch.Tensor) -> torch.Tensor:
        x = self.gen(latent_vec)
        return x

class Autoencoder(nn.Module):
    def __init__(self, im_chan: int = 1, z_dim: int = 64, hidden_dim: int = 32):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(im_chan, z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, im_chan, hidden_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(image) # (batch_size, z_dim, 1, 1)
        reconstructed = self.decoder(latent) # (batch_size, im_chan, 28, 28)
        return reconstructed