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

## Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

## Define hyperparameters
batch_size = 128
lr = 2e-4 
beta1 = 0.5 
beta2 = 0.999 
n_epochs = 50 
display_step = 500 
real_label_smoothing = 0.9 

z_dim = 64          
n_classes = 10      
gen_hidden_dim = 64 
disc_hidden_dim = 16 
mnist_shape = (1, 28, 28) 

gen_input_dim, disc_input_chan = get_input_dimensions(z_dim, mnist_shape, n_classes)

## Initialize models
gen= Generator(input_dim=gen_input_dim, z_dim=z_dim, hidden_dim=gen_hidden_dim).to(device)
disc = Discriminator(input_dim=disc_input_chan, hidden_dim=disc_hidden_dim, n_classes=n_classes).to(device)

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

        one_hot_labels_vec, image_one_hot_labels = get_one_hot_labels(labels, n_classes, mnist_shape, device)
        pass
