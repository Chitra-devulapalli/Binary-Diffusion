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
timesteps = [50, 40, 30, 20, 10, 0] 
num_samples = 10000

# Generate 10,000 samples
noisy_samples = torch.randn(num_samples, 2).to(device).unsqueeze(1)
# print(noisy_samples.shape)
# raise Exception

# Denoising function (reverse diffusion)
def denoise_samples(model, samples, time_horizon, y):
    denoised_samples_list = []
    for t_index in time_horizon:
        t = torch.full((samples.shape[0],), t_index, device=device, dtype=torch.long)
        samples = p_sample(model, samples, t, t_index, y)
        denoised_samples_list.append(samples.cpu().detach().numpy().squeeze(1))
    return denoised_samples_list


def plot_density_with_trajectory(trajectory1, trajectory2, ground_truth=None, save_dir="plots"):
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
    
    plt.contourf(X, Y, Z_gt, cmap='Blues', levels=50)

    #trajectory
    trajectory1 = np.array(trajectory1)
    trajectory2 = np.array(trajectory2)
    plt.plot(trajectory1[:, 0], trajectory1[:, 1], 'ro-', markersize=1, label = "1")
    plt.scatter(trajectory1[-1, 0], trajectory1[-1, 1], color='red',s=20)
    plt.plot(trajectory2[:, 0], trajectory2[:, 1], 'bo-', markersize=1, label = "0")
    plt.scatter(trajectory2[-1, 0], trajectory2[-1, 1], color='blue',s=20)


    plt.title(f"Denoising for y=0 and y=1")
    plt.legend(loc = "upper right")

    print("SAVING")
    plt.savefig((f"question4.png"))

time_horizon = list(range(50, -1, -1))
# Get denoised samples at each time step
print("STARTING DENOISING")
denoised_samples_list1 = denoise_samples(model, noisy_samples, time_horizon,y=0)
denoised_samples_list2 = denoise_samples(model, noisy_samples, time_horizon,y=1)
# print(len(denoised_samples_list))
# raise Exception
# Select a particular sample to track its trajectory
sample_index = np.random.randint(0, num_samples)
trajectory1 = np.array([denoised_samples[sample_index] for denoised_samples in denoised_samples_list1])
trajectory2 = np.array([denoised_samples[sample_index] for denoised_samples in denoised_samples_list2])
# print("SHAPE", trajectory.shape)
# trajectory=[]

# Plot the evolution of the sample distribution and the trajectory
for i, t in enumerate(timesteps):
    if t == 0:
        print(i)
        # raise Exception
        plot_density_with_trajectory(trajectory1[:(i*10)+1], trajectory2[:(i*10)+1], t)