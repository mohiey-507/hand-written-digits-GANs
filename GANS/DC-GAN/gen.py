import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.layers = nn.Sequential(
        # Input: (z_dim, 1,1) -> Output: (hidden_dim * 4, 3, 3)
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

        # Input: (hidden_dim, 14, 14) -> Output: (1, 28, 28)
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1,
                            kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def get_noise(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.layers(x)