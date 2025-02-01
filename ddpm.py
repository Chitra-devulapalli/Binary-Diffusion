import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None]


class UNet(nn.Module):
    """
    A time-dependent model built upon U-Net architecture.
    For this UNet, we will only have 2 layers for each pathways

    Encoding Layer 1 consist of a conv1d with strides 2, 

    Encoding: x -> Conv2d -> y + 
    """

    def __init__(self, channels=[32, 64], embed_dim=256):
        """Initialize a time-dependent unet.

        Args:
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        #CLASSIFIER FREE GUIDANCE
        self.label_embed = nn.Embedding(2,embed_dim)
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv1d(1, channels[0], kernel_size=2, stride=2)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=1, stride=1)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        # Decoding layers
        self.tconv2 = nn.ConvTranspose1d(channels[1], channels[0], kernel_size=1, stride=1)
        self.dense3 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(4, channels[0])

        self.tconv1 = nn.ConvTranspose1d(channels[1], 1, kernel_size=2, stride=1)

        # The swish activation function
        self.act = nn.SiLU()
        
    def forward(self, x, t, cond=None, y=None):
        # t_emb = self.act(self.embed(t))
        # if y is not None and (y != -1).all():
        #     y_emb = self.act(self.label_embed(y))
        #     t_emb = t_emb + y_emb

        t_emb = self.act(self.embed(t))
        if y is not None:
            y = y.to(x.device)
            y = y.long()  # Ensure y is of type long
            # Handle cases where y might contain -1 indicating invalid labels
            valid_mask = (y != -1)
            if valid_mask.any():
                y_emb = torch.zeros_like(t_emb)
                y_emb[valid_mask] = self.act(self.label_embed(y[valid_mask]))
                t_emb = t_emb + y_emb

        # Encoding path
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(t_emb)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(t_emb)))

        # Decoding path
        h3 = self.act(self.tgnorm2(self.tconv2(h2) + self.dense3(t_emb)))
        h4 = torch.concat([h1,h3], dim=1)
        out = self.tconv1(h4)
        return out