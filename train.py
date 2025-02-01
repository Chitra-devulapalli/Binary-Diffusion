import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import trange
from torch.optim import Adam
from scipy.stats import gaussian_kde

from ddpm import UNet
from gmm import GMM
from utils import *
torch.manual_seed(3)

device = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    model = UNet()
    model.to(device)

    n_epochs =   100
    batch_size =  64
    lr=5e-4
    timesteps = 50
    cosine_beta_schedule(timesteps, s=0.008)

    means = torch.tensor([[-0.35, 0.65], [0.75, -0.45]])
    std_devs = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
    mix_weights = torch.tensor([.35, .65])
    dataset = GMM(means, std_devs, mix_weights)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4) #can add transforms for normalization

    optimizer = Adam(model.parameters(), lr=lr)

    tqdm_epoch = trange(n_epochs)
    for _ in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x,y in data_loader:
            print("Y", y)
            x,y = x.to(device), y.to(device)
            t = torch.randint(0, timesteps, (x.shape[0],), device=device).long()
            #CLASSIFIER FREE GUIDANCE
            p_uncond = 0.1
            mask = torch.rand(y.shape, device=device) < p_uncond
            y[mask] = -1
            
            loss = p_losses(model, x, t, y=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        torch.save(model.state_dict(), '50_timesteps.pth')
        
if __name__=="__main__":
    train()