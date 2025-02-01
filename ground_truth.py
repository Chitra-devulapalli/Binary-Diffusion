import torch
import numpy as np
import matplotlib.pyplot as plt
from gmm import GMM
from torch.distributions import Categorical, Normal, MixtureSameFamily, Independent

# GMM Class (same as provided)
class GMM:
    def __init__(self, means, std_devs, mix_weights):
        mix_weights = Categorical(mix_weights)
        component_distribution = Independent(Normal(means, std_devs), 1)
        gmm = MixtureSameFamily(mix_weights, component_distribution)
        samples = []
        l = 0
        while l < 10000:
            new_samples = gmm.sample((1000,))
            crit1 = -1.5 < new_samples[:, 0]
            crit2 = new_samples[:, 0] < 1.5
            crit3 = -1.5 < new_samples[:, 1]
            crit4 = new_samples[:, 1] < 1.5
            crit = crit1 & crit2 & crit3 & crit4
            samples.append(new_samples[crit])
            l += len(new_samples[crit])
        self.samples = torch.cat(samples, dim=0)

# Define GMM parameters
means = torch.tensor([[-0.35, 0.65], [0.75, -0.45]])
std_devs = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
mix_weights = torch.tensor([.35, .65])

# Generate samples
gmm = GMM(means, std_devs, mix_weights)
samples = gmm.samples.numpy()

# Plotting the GMM distribution using plt.contourf
x_samples, y_samples = samples[:, 0], samples[:, 1]

# Create a grid for contour plot
x_min, x_max = -4.0 , 4.0
y_min, y_max = -4.0 , 4.0
x_grid, y_grid = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x_grid, y_grid)

# Kernel Density Estimation for contour plot
from scipy.stats import gaussian_kde
xy = np.vstack([x_samples, y_samples])
kde = gaussian_kde(xy)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# Plot the 2D density
plt.contourf(X, Y, Z, cmap='Blues', levels=50)
plt.title("Ground Truth")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Density")
plt.savefig("Ground Truth.png")