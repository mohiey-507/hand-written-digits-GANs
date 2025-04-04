import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor, num_images=25):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_input_dimensions(noise_dim, image_shape=(1, 28, 28), num_classes=10):
    """Calculates input dimensions for cGAN components."""
    generator_input_dim = noise_dim + num_classes
    discriminator_input_channels = image_shape[0] + num_classes
    return generator_input_dim, discriminator_input_channels

def concat_vectors(x, y):
    """Concatenates two tensors along the channel dimension (dim=1)."""
    return torch.cat((x, y), dim=1).float()

def get_one_hot_labels(labels, n_classes=10, image_shape=(1, 28, 28), device='cpu'):
    """
    Generates one-hot labels suitable for concatenating with noise or images.

    Returns:
        one_hot_labels_vec: Shape (batch_size, n_classes, 1, 1) for Generator.
        one_hot_labels_map: Shape (batch_size, n_classes, H, W) for Discriminator.
    """
    # Create basic one-hot tensor: (batch_size, n_classes)
    one_hot_labels = F.one_hot(labels, num_classes=n_classes).to(device).float()
    # Reshape for noise concatenation: (batch_size, n_classes, 1, 1)
    one_hot_labels_vec = one_hot_labels.unsqueeze(-1).unsqueeze(-1)
    # Reshape for image concatenation: (batch_size, n_classes, H, W)
    one_hot_labels_map = one_hot_labels_vec.repeat(1, 1, image_shape[1], image_shape[2])
    return one_hot_labels_vec, one_hot_labels_map

def visualize_all_digits(generator, device, n_classes=10, examples_per_digit=9):
    """
    Generates and visualizes examples for each digit (0-9) using the generator.

    Displays a 2x5 grid, where each cell corresponds to a digit.
    Inside each cell, a smaller grid (e.g., 3x3) shows multiple generated
    examples of that specific digit.

    Args:
        generator (nn.Module): The trained Generator model.
        device (torch.device): The device (CPU/GPU) to run generation on.
        z_dim (int): The dimension of the noise vector input to the generator.
        n_classes (int): The total number of classes (digits). Default is 10.
        examples_per_digit (int): The number of examples to generate for each digit.
                                Should ideally be a perfect square (e.g., 9 for 3x3).
                                Default is 9.
    """
    generator.eval()

    # Determine the layout of the examples within each digit's subplot
    inner_grid_rows = int(math.sqrt(examples_per_digit))
    inner_grid_cols = int(math.ceil(examples_per_digit / inner_grid_rows))
    if inner_grid_rows * inner_grid_cols != examples_per_digit:
        print(f"Warning: examples_per_digit ({examples_per_digit}) is not a perfect square. "
               f"Using inner grid size {inner_grid_rows}x{inner_grid_cols}.")

    # Create the main 2x5 figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle(f'{examples_per_digit} Generated Examples for Each Digit (0-9)', fontsize=16)

    with torch.no_grad(): 
        for digit in range(n_classes):

            # Create noise vectors and labels for the current digit
            noise = generator.get_noise(examples_per_digit, device=device)
            labels = torch.full((examples_per_digit,), digit, dtype=torch.long, device=device)

            one_hot_labels_vec, _ = get_one_hot_labels(
                labels, n_classes=n_classes, device=device
            )

            generator_input = concat_vectors(noise, one_hot_labels_vec)

            generated_images = generator(generator_input)

            # Denormalize images
            generated_images = (generated_images + 1) / 2
            generated_images = generated_images.clamp(0, 1) # Ensure values are in [0, 1]

            # Create the inner grid (e.g., 3x3) for the current digit
            image_grid = make_grid(generated_images, nrow=inner_grid_cols, padding=2, normalize=False)

            # Determine the position in the main 2x5 grid
            main_row, main_col = divmod(digit, 5)
            ax = axes[main_row, main_col]

            # Display the grid image
            ax.imshow(image_grid.permute(1, 2, 0).cpu().squeeze(), cmap='gray')
            ax.set_title(f"Digit: {digit}")
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    generator.train() 