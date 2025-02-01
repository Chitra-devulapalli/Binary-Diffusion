import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from gmm import GMM

from ddpm import UNet
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet()
model.load_state_dict(torch.load('50_timesteps.pth'))
model.to(device)
model.eval()

#running cosine beta schedule
timesteps_schedule = 51
cosine_beta_schedule(timesteps_schedule, s=0.008)

# Set time steps
timesteps = [0,10,20,30,40,50] 
num_samples = 10000

# Generate 10,000 samples
noisy_samples = torch.randn(num_samples, 2).to(device).unsqueeze(1)
# print(noisy_samples.shape)
# raise Exception

# Denoising function (reverse diffusion)
def denoise_samples(model, samples, time_horizon):
    denoised_samples_list = []
    for t_index in time_horizon:
        t = torch.full((samples.shape[0],), t_index, device=device, dtype=torch.long)
        samples = p_sample(model, samples, t, t_index)
        denoised_samples_list.append(samples.cpu().detach().numpy().squeeze(1))
    return denoised_samples_list


def plot_density_with_trajectory(denoised_samples, trajectory, step, ax, ground_truth=None, save_dir="plots"):
    x_samples, y_samples = denoised_samples.T

    #kernel density estimation (KDE)
    xy = np.vstack([x_samples, y_samples])
    kde = gaussian_kde(xy)
    
    #Define grid
    x_min, x_max = -3.0 , 3.0
    y_min, y_max = -3.0 , 3.0
    x_grid, y_grid = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    ax.contourf(X, Y, Z, cmap='Blues', levels=50)

    #trajectory
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=1)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red',s=20)

    ax.set_title(f"Denoising at t={step}")
    # ax.legend()

    # print("SAVING")
    # plt.savefig((f"pv1_guidance_t_{step}.png"))

time_horizon = list(range(50, -1, -1))
# Get denoised samples at each time step
denoised_samples_list = denoise_samples(model, noisy_samples, time_horizon)

# Select a particular sample to track its trajectory
sample_index = np.random.randint(0, num_samples)
trajectory = np.array([denoised_samples[sample_index] for denoised_samples in denoised_samples_list])

n_plots = len(timesteps) + 1  # Add 1 for ground truth
fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

# Plot the evolution of the sample distribution and the trajectory
for i, t in enumerate(timesteps):
    plot_density_with_trajectory(denoised_samples_list[i*10], trajectory[:(i*10)+1], t, axes[i])


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
Z = kde_gt(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# Plot ground truth
axes[-1].contourf(X, Y, Z, cmap='Blues', levels=50)
axes[-1].set_title("Ground Truth")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("question2_new.png")