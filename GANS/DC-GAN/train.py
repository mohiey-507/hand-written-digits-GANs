## Import libraries
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from utils import show_tensor_images, weights_init, plot_captured_images
from disc import Discriminator
from gen import Generator

import numpy as np
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
gen_hidden_dim = 128 
desc_hidden_dim = 32 
z_dim = 100
batch_size = 128

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
n_epochs = 50
display_step = 1000
real_label_smoothing = 0.9

## Load MNIST dataset
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

## Define discriminator and generator
gen = Generator(z_dim, hidden_dim=gen_hidden_dim).to(device)
disc = Discriminator(hidden_dim=desc_hidden_dim).to(device) 

print("Initializing weights...")
gen.apply(weights_init)
disc.apply(weights_init)
print("Weights initialized.")

## Define optimizer and loss function
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))

criterion = nn.BCEWithLogitsLoss()

# Put models in train mode
disc.train()
gen.train()

# Determine epochs for capture
if n_epochs <= 10:
    target_capture_epochs = set(range(1, n_epochs + 1))
else:
    capture_indices = np.linspace(0, n_epochs - 1, 10, dtype=int)
    target_capture_epochs = set(idx + 1 for idx in capture_indices)

cur_step = 0
captured_images_data = []

print("Starting Training Loop...")
for epoch in range(n_epochs):

    total_gen_loss_epoch = 0.0
    total_disc_loss_epoch = 0.0
    epoch_num = epoch + 1

    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num}/{n_epochs}")
    for real_images, _ in pbar:
        real_images = real_images.to(device)

        # -------------------------
        #  Train Discriminator
        # -------------------------
        disc_opt.zero_grad()

        # Generate fake images
        noise = gen.get_noise(batch_size, device=device)
        fake_images = gen(noise)

        real_logits = disc(real_images)
        fake_logits_d = disc(fake_images.detach())

        # Calculate Discriminator loss
        # Real images loss (use label smoothing)
        real_targets = (torch.ones_like(real_logits) * real_label_smoothing).to(device)
        real_loss = criterion(real_logits, real_targets)

        # Fake images loss (target is 0)
        fake_targets_d = torch.zeros_like(fake_logits_d).to(device)
        fake_loss_d = criterion(fake_logits_d, fake_targets_d)

        # Combine losses
        disc_loss = (real_loss + fake_loss_d) / 2

        # Backpropagate and update Discriminator
        disc_loss.backward()
        disc_opt.step()

        total_disc_loss_epoch += disc_loss.item()

        # ---------------------
        #  Train Generator
        # ---------------------
        gen_opt.zero_grad()

        fake_logits_g = disc(fake_images)

        # Calculate Generator loss - Generator wants Discriminator to output 1 (real) for fake images
        targets_g = torch.ones_like(fake_logits_g).to(device) # Target is 1
        gen_loss = criterion(fake_logits_g, targets_g)

        # Backpropagate and update Generator
        gen_loss.backward()
        gen_opt.step()

        total_gen_loss_epoch += gen_loss.item()

        # Update progress bar description with current batch losses
        pbar.set_postfix({
            "D Loss": f"{disc_loss.item():.4f}",
            "G Loss": f"{gen_loss.item():.4f}"
        })

        # -------------------------
        #  Step-based Visualization
        # -------------------------
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"\n-- Step {cur_step} / Epoch {epoch_num} --")
            print(f"  Generator Loss (step): {gen_loss.item():.4f}")
            print(f"  Discriminator Loss (step): {disc_loss.item():.4f}")

            print("  Generated Images:")
            gen.eval()
            with torch.no_grad():
                noise_vis = gen.get_noise(25, device=device)
                vis_images = gen(noise_vis)
                show_tensor_images(vis_images)
            gen.train()

        cur_step += 1

    # --- End of Epoch ---
    # Calculate average losses for the completed epoch
    avg_disc_loss_epoch = total_disc_loss_epoch / len(dataloader)
    avg_gen_loss_epoch = total_gen_loss_epoch / len(dataloader)
    print("-" * 40)
    print(f"Epoch {epoch+1} Completed:")
    print(f"  Avg Generator Loss: {avg_gen_loss_epoch:.4f}")
    print(f"  Avg Discriminator Loss: {avg_disc_loss_epoch:.4f}")

    # -------------------------
    #  Epoch-based Image Capture
    # -------------------------
    if epoch_num in target_capture_epochs:
        print(f"  Capturing images at end of Epoch {epoch_num}...")
        gen.eval()
        with torch.no_grad():
            noise_capture = gen.get_noise(25, device=device)
            captured_images = gen(noise_capture)
        gen.train()

        captured_images_data.append((epoch_num, captured_images.detach().cpu()))
    print("-" * 40)

print("Training Finished.")
plot_captured_images(captured_images_data)