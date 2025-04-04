import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(Discriminator, self).__init__()

        # Adjust input channels of the first layer
        self.block1 = nn.Sequential(
            # input_dim = im_chan + n_classes
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        # Output shape: (hidden_dim, 14, 14)

        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False), # Adjusted padding for size
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2)
        )
        # Output shape: (hidden_dim*2, 7, 7)

        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, 1, kernel_size=7, stride=1, padding=0, bias=False) # Adjusted kernel_size
        )
        # Output shape: (1, 1, 1)

    def forward(self, image_and_labels):
        # image_and_labels shape is : (batch_size, input_dim, 28, 28)
        x = self.block1(image_and_labels)
        x = self.block2(x)
        x = self.block3(x)
        # Reshape to (batch_size, 1)
        return x.view(x.size(0), -1)