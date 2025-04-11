## Import libraries
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from model import Autoencoder
from utils import *

from tqdm import tqdm

## Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

## Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

## Define hyperparameters
batch_size = 128
hidden_dim = 16 
z_dim = 32

lr = 1e-3
n_epochs = 7
display_step = 500
n_examples_to_show = 5

## Load MNIST dataset
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

## Define Autoencoder, loss function, optimizer
ae = Autoencoder(z_dim=z_dim, hidden_dim=hidden_dim).to(device)
ae_opt = torch.optim.Adam(ae.parameters(), lr=lr)
criterion = nn.MSELoss()

epoch_losses_history = [] 
cur_step = 0
ae.train()

print("Starting Training Loop...")
for epoch in range(n_epochs):

    epoch_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
    for image, _ in pbar:
        image = image.to(device)

        # 1. Forward pass
        reconstructed_image = ae(image)

        # 2. Compute loss
        loss = criterion(reconstructed_image, image)
        cur_loss = loss.item()

        epoch_loss += cur_loss 

        # 3. Zero grad
        ae_opt.zero_grad()

        # 4. Backward pass
        loss.backward()

        # 5. Update weights
        ae_opt.step()

        # Update progress bar description with current batch losses
        pbar.set_postfix({
            "Loss": f"{cur_loss:.4f}",
        })

        ## Visualization
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"\n-- Step {cur_step} / Epoch {epoch+1} --")
            print(f"  Loss: {loss.item():.4f}")

            visualize_reconstructions(image, reconstructed_image, n_examples_to_show)

        cur_step += 1

    # --- End of Epoch ---
    avg_epoch_loss = epoch_loss / len(dataloader) 
    epoch_losses_history.append(avg_epoch_loss) 

    print("-" * 40)
    print(f"\n-- End of Epoch {epoch+1} --")
    print(f"  Average Epoch Loss: {avg_epoch_loss:.4f}")
    print("-" * 40)

print("Training Finished.")
visualize_reconstructions(image, reconstructed_image, grid_size=6)
plot_epoch_loss(epoch_losses_history)