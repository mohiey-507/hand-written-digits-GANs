import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, z_dim=64, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim 

        # Adjust input channels of the first layer to the combined dimension
        self.block1 = nn.Sequential(
            # input_dim = z_dim + n_classes
            nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim * 4, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True)
        )
        # Output shape: (hidden_dim*4, 3, 3)

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2,
                            kernel_size=4, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        # Output shape: (hidden_dim*2, 7, 7)

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim * 2, out_channels=hidden_dim,
                            kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # Output shape: (hidden_dim, 14, 14)

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1, # Output 1 channel for MNIST
                            kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() # Output range [-1, 1]
        )
        # Output shape: (1, 28, 28)

    def get_noise(self, batch_size, device='cpu'):
        return torch.randn(batch_size, self.z_dim, 1, 1, device=device)

    def forward(self, noise_and_labels):
        x = self.block1(noise_and_labels)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x