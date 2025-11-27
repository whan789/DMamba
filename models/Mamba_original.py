import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models import Mamba


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ResidualMambaBlock(nn.Module):
    def __init__(self, mamba_block, d_model):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = mamba_block

    def forward(self, x):
        return self.mamba(self.norm(x)) + x


class Model(nn.Module):
    def __init__(self, configs, mamba_class):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # Embedding
        from layers.Embed import DataEmbedding
        self.embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Mamba Layers
        self.layers = nn.ModuleList([
            ResidualMambaBlock(
                mamba_class(
                    d_model=configs.d_model,
                    d_state=configs.d_state,
                    d_conv=configs.d_conv,
                    expand=configs.expand,
                    layer_idx=i,
                    use_fast_path=True
                ),
                d_model=configs.d_model
            ) for i in range(configs.e_layers)
        ])

        self.norm = RMSNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - mean_enc) / std_enc

        x_enc = self.embedding(x_enc, x_mark_enc)  # [B, L, D]

        for layer in self.layers:
            x_enc = layer(x_enc)
        x_enc = self.norm(x_enc)

        x_out = self.projection(x_enc)
        x_out = x_out * std_enc + mean_enc

        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["short_term_forecast", "long_term_forecast"]:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
