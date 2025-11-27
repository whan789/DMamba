import torch

def norm_in(x):
    mean = x.mean(1, keepdim=True)
    std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return (x - mean) / std, mean, std