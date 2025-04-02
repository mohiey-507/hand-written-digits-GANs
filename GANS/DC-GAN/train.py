## Import libraries
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from utils import show_tensor_images
from disc import Discriminator
from gen import Generator
from loss import GenLoss, DiscLoss

from tqdm import tqdm

## Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

## Define device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

## Define hyperparameters
batch_size = 128
lr = 2e-5
n_epochs = 50
z_dim = 64
display_step = 2000

## Load MNIST dataset
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

## Define discriminator and generator
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

## Define loss function
gen_loss_fn = GenLoss()
disc_loss_fn = DiscLoss()

def train(
    dataloader: DataLoader,
    disc: nn.Module,
    gen: nn.Module,
    disc_loss_fn: nn.Module,
    gen_loss_fn: nn.Module,
    disc_opt: optim.Optimizer,
    gen_opt: optim.Optimizer,
    device: torch.device=device,
    n_epochs: int=n_epochs,
    z_dim: int=z_dim,
    display_step: int=display_step,
    show_tensor_images=show_tensor_images,
):
    disc.train()
    gen.train()
    cur_step = 0
    for epoch in range(n_epochs):
        total_gen_loss, total_disc_loss = 0, 0

        for real, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            real = real.to(device)

            # -------------------------
            # 1. Update Discriminator
            # -------------------------
            # Forward pass on real images
            real_logits = disc(real)
            # Generate fake images and forward pass
            noise = gen.get_noise(real.shape[0], z_dim, device=device)
            fake_images = gen(noise)
            fake_logits = disc(fake_images.detach())

            # Calculate discriminator loss and update
            disc_loss = disc_loss_fn(real_logits, fake_logits)
            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()
            total_disc_loss += disc_loss.item()

            # -------------------------
            # 2. Update Generator
            # -------------------------
            noise = gen.get_noise(real.shape[0], z_dim, device=device)
            fake_images = gen(noise)
            fake_logits_for_gen = disc(fake_images)

            # Calculate generator loss and update
            gen_loss = gen_loss_fn(fake_logits_for_gen)
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()
            total_gen_loss += gen_loss.item()

            # -------------------------
            # 3. Visualization (if enabled)
            # -------------------------
            if cur_step % display_step == 0 and cur_step > 0:
                mean_generator_loss = total_gen_loss / (cur_step + 1)
                mean_discriminator_loss = total_disc_loss / (cur_step + 1)
                print(f"Epoch {epoch+1}, step {cur_step}: Generator loss: {mean_generator_loss:.4f}, Discriminator loss: {mean_discriminator_loss:.4f}")
                show_tensor_images(fake_images)
                show_tensor_images(real)

            cur_step += 1

        # Average losses for the epoch
        avg_disc_loss = total_disc_loss / len(dataloader)
        avg_gen_loss = total_gen_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed: Avg Generator Loss: {avg_gen_loss:.4f}, Avg Discriminator Loss: {avg_disc_loss:.4f}")

    return avg_disc_loss, avg_gen_loss

train(dataloader, disc, gen, disc_loss_fn, gen_loss_fn, disc_opt, gen_opt, device, n_epochs, z_dim, display_step, show_tensor_images)