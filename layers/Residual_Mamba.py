import torch
import torch.nn as nn
from layers.RMSNorm import RMSNorm

class ResidualMambaBlock(nn.Module):
    def __init__(self, mamba_block, d_model):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = mamba_block

    def forward(self, x):
        return self.mamba(self.norm(x)) + x