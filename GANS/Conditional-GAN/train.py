## Import libraries
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from models import Generator, Discriminator
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
lr = 2e-4 
beta1 = 0.5 
beta2 = 0.999 
n_epochs = 50 
display_step = 500 
real_label_smoothing = 0.9 

z_dim = 90          
n_classes = 10      
gen_hidden_dim = 128 
disc_hidden_dim = 32 
mnist_shape = (1, 28, 28) 

gen_input_dim, disc_input_chan = get_input_dimensions(z_dim, mnist_shape, n_classes)

## Initialize models
gen= Generator(input_dim=gen_input_dim, z_dim=z_dim, hidden_dim=gen_hidden_dim).to(device)
disc = Discriminator(input_dim=disc_input_chan, hidden_dim=disc_hidden_dim).to(device)

## Load MNIST dataset
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

## Initialize weights
gen.apply(weights_init)
disc.apply(weights_init)
print("Weights initialized.")

## Define optimizer and loss function
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))

# Use BCEWithLogitsLoss for stability (handles sigmoid internally)
criterion = nn.BCEWithLogitsLoss()

## Train the model
disc.train()
gen.train()

cur_step = 0
print("Starting Training Loop...")
for epoch in range(n_epochs):

    total_gen_loss_epoch = 0.0
    total_disc_loss_epoch = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=True)
    for real_images, labels in pbar:
        real_images, labels = real_images.to(device), labels.to(device)

        one_hot_labels_vec = F.one_hot(labels, num_classes=n_classes).to(device).float()
        image_one_hot_labels = one_hot_labels_vec[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

        # -------------------------
        #  Train Discriminator
        # -------------------------

        disc_opt.zero_grad()

        # Generate fake images
        fake_noise = gen.get_noise(batch_size, device=device)
        # Concatenate noise and vector labels for Generator input
        noise_and_labels = concat_vectors(fake_noise, one_hot_labels_vec)
        # Generate fake images
        fake_images = gen(noise_and_labels)

        # Combine images with label maps for Discriminator input
        fake_image_and_labels = concat_vectors(fake_images.detach(), image_one_hot_labels)
        real_image_and_labels = concat_vectors(real_images, image_one_hot_labels)

        # Get discriminator predictions 
        real_logits = disc(real_image_and_labels)
        fake_logits_d = disc(fake_image_and_labels)

        # Calculate Discriminator loss - applying smoothing to real labels
        real_targets = (torch.ones_like(real_logits) * real_label_smoothing).to(device)
        disc_real_loss = criterion(real_logits, real_targets)

        # Targets for fake images are zeros
        fake_targets_d = torch.zeros_like(fake_logits_d).to(device)
        disc_fake_loss = criterion(fake_logits_d, fake_targets_d)

        # Average the real and fake loss
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Backpropagate and update Discriminator
        disc_loss.backward()
        disc_opt.step()

        total_disc_loss_epoch += disc_loss.item()

        # -------------------------
        #  Train Generator
        # -------------------------

        gen_opt.zero_grad()

        # Get discriminator predictions for the *Generator's* output
        fake_image_and_labels_g = concat_vectors(fake_images, image_one_hot_labels)
        fake_logits_g = disc(fake_image_and_labels_g)

        # Calculate Generator loss - Generator wants Discriminator to predict 1 (real)
        targets_g = torch.ones_like(fake_logits_g).to(device)
        gen_loss = criterion(fake_logits_g, targets_g)

        # Backpropagate and update Generator
        gen_loss.backward()
        gen_opt.step()

        total_gen_loss_epoch += gen_loss.item()

        # Update progress bar description with current batch losses
        pbar.set_postfix({
            "D Loss": f"{disc_loss.item():.6f}",
            "G Loss": f"{gen_loss.item():.6f}"
        })

        # -------------------------
        #  Visualization
        # -------------------------
        if cur_step % display_step == 0 and cur_step > 0:
            pbar.write(f"\n-- Step {cur_step} --") 
            pbar.write(f"  Generator Loss (step): {gen_loss.item():.4f}")
            pbar.write(f"  Discriminator Loss (step): {disc_loss.item():.4f}")

            pbar.write("  Generated Images:")
            gen.eval()

            num_vis = 25 # Number of images to visualize
            vis_labels = torch.randint(0, 10, (num_vis,), device=device).long()

            # Get noise for visualization
            noise_vis = gen.get_noise(num_vis, device=device)
            # Get one-hot labels (vector format needed for Gen input)
            one_hot_labels_vis_vec = F.one_hot(vis_labels, num_classes=n_classes).to(device).float()
            # Concatenate for generator
            noise_and_labels_vis = concat_vectors(noise_vis, one_hot_labels_vis_vec)

            with torch.no_grad():
                vis_images = gen(noise_and_labels_vis)
            show_tensor_images(vis_images, num_images=num_vis)

            gen.train()

        cur_step += 1

    # --- End of Epoch ---
    avg_disc_loss_epoch = total_disc_loss_epoch / len(dataloader)
    avg_gen_loss_epoch = total_gen_loss_epoch / len(dataloader)
    print("-" * 40)
    print(f"Epoch {epoch+1} Completed:")
    print(f"  Avg Generator Loss: {avg_gen_loss_epoch:.5f}")
    print(f"  Avg Discriminator Loss: {avg_disc_loss_epoch:.5f}")
    print("-" * 40)
print("Training Finished.")