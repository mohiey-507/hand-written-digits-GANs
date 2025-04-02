import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        
        # Input: (input_dim, 28, 28)
        self.block1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # From (hidden_dim, 13, 13) -> (hidden_dim*2, 5, 5)
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2)
        )

        # From (hidden_dim*2, 5, 5) -> (1, 1, 1)
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, 1, kernel_size=5, stride=1)
        )

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        x = x.view(x.size(0), -1) # output shap is (batch_size, 1)
        return x