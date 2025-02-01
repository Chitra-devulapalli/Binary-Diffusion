import numpy as np

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch import nn, einsum
import torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    global betas                    
    global alphas
    global alphas_cumprod                   # alpha_bar
    global posterior_variance
    global max_timesteps
    
    max_timesteps = timesteps
    #TODO: finish the other array
    t = np.linspace(0, timesteps, timesteps + 1) / timesteps
    
    #Cosine Schedule
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize so alpha_bar[0] = 1

    #Clipping alphas 
    alphas_cumprod = np.clip(alphas_cumprod,1e-4,0.999)

    #Compute betas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    #Clipping betas
    betas = np.clip(betas,1e-4,0.999)

    #Compute alphas
    alphas = 1 - betas
    
    #posterior variance
    posterior_variance = betas * (1 - alphas_cumprod[:-1]) / (1 - alphas_cumprod[1:])
    
    #variables as  tensors
    betas = torch.tensor(betas, dtype=torch.float32)
    alphas = torch.tensor(alphas, dtype=torch.float32)
    alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
    posterior_variance = torch.tensor(posterior_variance, dtype=torch.float32)

def p_losses(denoise_model, x_start, t, y=None, w=0.5):
    """
    Calculate loss between actual noise and predicted noise at time step t.
    
    Args:
        denoise_model: The UNet model that predicts the noise.
        x_start: The original image (x_0).
        t: The current time step.
    
    Returns:
        loss: The training loss (MSE).
    """
    # Sample noise to add to x_start
    noise = torch.randn_like(x_start)
    # print('Noise', noise)
    sqrt_alpha_cumprod_t = alphas_cumprod[t].view(-1, 1, 1) ** 0.5  # Reshape to [batch_size, 1, 1]
    sqrt_one_minus_alpha_cumprod_t = (1 - alphas_cumprod[t]).view(-1, 1, 1) ** 0.5
    
    # print("x_start",x_start.shape)
    # print("sqrt", sqrt_alpha_cumprod_t.shape)
    #Noisy image at t
    x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
    # print(y)
    # raise Exception 
    predicted_noise = denoise_model(x_t,t,y=y)
    # Loss between actual noise and predicted noise (we use L2 loss, i.e., MSE)
    loss = F.smooth_l1_loss(predicted_noise, noise, beta = 1.0)
    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index, y=None, w=0.5):
    """
    Perform a single reverse process step to sample from p(x_t-1 | x_t).
    
    Args:
        model: The denoising model (UNet).
        x: The noisy image at step t (x_t).
        t: The current time step (tensor of shape [batch_size]).
        t_index: The current time step index.
    
    Returns:
        x_t-1: The denoised image at time step t-1.
    """

    if y is not None and not torch.is_tensor(y):
        y = torch.full((x.shape[0],), y, device=x.device, dtype=torch.long)

    # Predict noise at step t
    predicted_noise_unconditional = model(x, t,y=None)
    predicted_noise_conditional = model(x,t,y=y)

    predicted_noise = (1 + w) * predicted_noise_conditional - w * predicted_noise_unconditional
    # print(predicted_noise)
    # raise Exception

    # Compute the coefficients needed for reverse process
    sqrt_alpha_t = alphas[t_index] ** 0.5
    sqrt_one_minus_alpha_t = (1 - alphas_cumprod[t_index]) ** 0.5
    beta_t = betas[t_index]
    pv_t = posterior_variance[t_index]
    
    # Compute x_t-1 using the reverse diffusion formula
    x_t_minus_1 = (1 / sqrt_alpha_t) * (x - (1 - alphas[t_index]) / sqrt_one_minus_alpha_t * predicted_noise)
    
    # Add some noise for sampling, except for the last time step
    if t_index > 1:
        z = torch.randn_like(x)  # Standard normal noise
        x_t_minus_1 += torch.sqrt(pv_t)* z
    
    return x_t_minus_1

# Returning all images
@torch.no_grad()
def p_sample_loop(model, shape, timesteps, y=None, w=0.5):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    imgs.append(img.cpu().numpy())

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, y=y, w=w)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, timesteps, batch_size=16, channels=3, y=None, w=0.5):
    return p_sample_loop(model, (batch_size, channels, *image_size), timesteps, y=y, w=w)
