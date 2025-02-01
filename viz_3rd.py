import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from gmm import GMM

from ddpm import UNet
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


#running cosine beta schedule
timesteps_schedule = 51
cosine_beta_schedule(timesteps_schedule, s=0.008)


def sample_and_denoise(T, model_path):
    # Initialize the model and diffusion process
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
        
    num_samples = 5000
    # Start from random noise
    samples = torch.randn(num_samples, 2).to(device)
    samples = samples.unsqueeze(1)
    # Denoise samples
    for t in reversed(range(T)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            samples = p_sample(model, samples, t_tensor,t)
    
    samples = samples.cpu().numpy()
    return samples

def plot_density(samples, ax, title):
    x_samples, y_samples = samples[:, 0], samples[:, 1]
    # Kernel Density Estimation
    xy = np.vstack([x_samples, y_samples])
    kde = gaussian_kde(xy)
    
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    x_grid, y_grid = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    contour = ax.contourf(X, Y, Z, cmap='Reds', levels=50)
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

def plot_samples_over_ground_truth(samples, ax, title):
    # Plot ground truth PDF
    # write a fxn to get groundtruth pdf
    means = torch.tensor([[-0.35, 0.65], [0.75, -0.45]])
    std_devs = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
    mix_weights = torch.tensor([.35, .65])
    gmm = GMM(means, std_devs, mix_weights)
    ground_truth_samples = gmm.samples.numpy()
    x_gt, y_gt = ground_truth_samples[:, 0], ground_truth_samples[:, 1]
    xy_gt = np.vstack([x_gt, y_gt])
    kde_gt = gaussian_kde(xy_gt)
    
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    x_grid, y_grid = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z_gt = kde_gt(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    ax.contourf(X, Y, Z_gt, cmap='Blues', levels=50)
    # Overlay denoised samples
    ax.scatter(samples[:, 0], samples[:, 1], color='red', s=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

def main():
    T_values = [5, 10, 25, 50]
    fig, axes = plt.subplots(len(T_values), 2, figsize=(10, 5 * len(T_values)))
    
    for idx, T in enumerate(T_values):
        print(f"Processing T={T}")
        model_path = f"{T}_timesteps.pth"
        samples = sample_and_denoise(T, model_path)
        samples = np.squeeze(samples)

        
        # First column: Samples over ground truth
        ax1 = axes[idx, 0]
        plot_samples_over_ground_truth(samples, ax1, title=f"T={T}: Samples over Ground Truth")
        
        # Second column: Estimated PDF from denoised samples
        ax2 = axes[idx, 1]
        plot_density(samples, ax2, title=f"T={T}: Estimated PDF from Denoised Samples")
    
    plt.tight_layout()
    plt.savefig("question3.png")
    plt.show()

if __name__ == "__main__":
    main()