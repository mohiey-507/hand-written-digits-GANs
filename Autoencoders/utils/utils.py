import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def visualize_reconstructions(original: torch.Tensor, reconstructed: torch.Tensor,
                            step: int, epoch: int, grid_size: int = 5):
    """
    Visualizes original and reconstructed images in a 1x2 figure, 
    where each side is a grid (grid_size x grid_size).
    """
    
    # Ensure we don't request more examples than available in the batch
    num_images = min(original.size(0), grid_size * grid_size)

    # Detach tensors, move to CPU, select images
    original_cpu = original.detach().cpu()[:num_images]
    reconstructed_cpu = reconstructed.detach().cpu()[:num_images]

    # --- Denormalization ---
    original_cpu = (original_cpu + 1 ) / 2 
    
    # Clamp values to [0, 1] range for display
    original_cpu = torch.clamp(original_cpu, 0, 1)
    reconstructed_cpu = torch.clamp(reconstructed_cpu, 0, 1)

    # --- Create Grids ---
    # Create a grid_size x grid_size grid for originals
    grid_original = make_grid(original_cpu, nrow=grid_size, padding=2) 
    # Create a grid_size x grid_size grid for reconstructed
    grid_reconstructed = make_grid(reconstructed_cpu, nrow=grid_size, padding=2)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(grid_size * 1.5, grid_size * 0.8))

    # Plot Original Images (Left Grid)
    axes[0].imshow(grid_original.permute(1, 2, 0).squeeze(), cmap='gray')
    axes[0].set_title(f'Original Images ({grid_size}x{grid_size})')
    axes[0].axis('off') 

    # Plot Reconstructed Images (Right Grid)
    axes[1].imshow(grid_reconstructed.permute(1, 2, 0).squeeze(), cmap='gray')
    axes[1].set_title(f'Reconstructed Images ({grid_size}x{grid_size})')
    axes[1].axis('off') 

    fig.suptitle(f'Epoch {epoch+1}, Step {step} Comparison', fontsize=12, y=.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()