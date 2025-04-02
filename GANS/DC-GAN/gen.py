import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=hidden_dim * 4, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True)
        )
        # From (z_dim, 1,1) -> (hidden_dim*4, 3,3)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2,
                            kernel_size=4, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        # From (hidden_dim*4, 3,3) -> (hidden_dim*2, 7,7)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim * 2, out_channels=hidden_dim,
                            kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # From (hidden_dim*2, 7,7) -> (hidden_dim, 14,14)
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1,
                            kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        # From (hidden_dim, 14,14) -> (1, 28,28)

    def get_noise(self, batch_size, z_dim=None, device='cpu'):
        if z_dim is None:
            z_dim = self.z_dim
        return torch.randn(batch_size, z_dim, 1, 1, device=device)

    def forward(self, x):
        return self.block4(self.block3(self.block2(self.block1(x))))