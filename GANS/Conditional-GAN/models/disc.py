import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 11, hidden_dim: int = 32):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
        # Input: (input_dim, 1, 1) -> Output: (hidden_dim, 14, 14)
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim,
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

        # Input: (hidden_dim * 4, 3, 3) -> Output: (1, 1, 1)
            nn.Conv2d(in_channels=hidden_dim * 4, out_channels=1,
                    kernel_size=3, stride=1, padding=0, bias=False)
        )

    def forward(self, image_and_labels):
        x = self.layers(image_and_labels)
        return x.view(x.size(0), -1)