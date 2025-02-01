import torch
from torch.distributions import Categorical, Normal, MixtureSameFamily, Independent
from torch.utils.data import Dataset

class GMM(Dataset):
    def __init__(self, means, std_devs, mix_weights):
        mix_weights = Categorical(mix_weights)
        component_distribution = Independent(Normal(means, std_devs), 1)
        gmm = MixtureSameFamily(mix_weights, component_distribution)
        samples = []
        labels = []
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
            #CLASSIFIER FREE GUIDANCE
            labels.append((new_samples[crit, 0] < 0).int())

        self.samples = torch.cat(samples, dim=0)
        self.labels = torch.cat(labels, dim=0)

    def __getitem__(self, index):
        # return self.samples[index][None, :], self.labels[index] #Uncomment for training
        return self.samples[index][None, :] # Uncomment for visualization
    
    def __len__(self):
        return len(self.samples)