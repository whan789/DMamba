import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from layers.Residual_Mamba import ResidualMambaBlock
from layers.RMSNorm import RMSNorm
from layers.STGAT_layer import GAT_TCN
from layers.Pretraining_module import TransformerDecoder, DecoderBlock, MambaDecoder
from layers.VarDrop import efficient_sampler
from utils.cal_corr import cal_corr_matrix, cal_spearman

class Model(nn.Module):
    def __init__(self, configs, mamba_class, dropout=0.1):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.embed = configs.embed
        self.freq = configs.freq
        self.dropout = dropout
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.use_vardrop = configs.use_vardrop
        self.k = configs.vardrop_k
        self.group_size = configs.group_size
        self.num_layers = configs.e_layers
        rfft_len = (self.seq_len) // 2 + 1
        lpf_cutoff = min(25, rfft_len // 2) 
        self.freq_list = list(range(lpf_cutoff))

        self.mamba_linear = nn.Linear(self.enc_in, self.d_model)
        self.d2p = nn.Linear(self.d_model, self.patch_len)
        self.output_linear = nn.Linear(self.seq_len, self.pred_len)
        self.patch_emb = nn.Linear(self.patch_len, self.d_model)

        # ====== Pretraining Args ======
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.remask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.dec_dim = self.d_model
        self.dec_depth = 2
        self.dec_nhead = 8
        self.ar_token = nn.Parameter(torch.zeros(1,1, self.dec_dim))
        # self.dec_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.dec_dim))
        self.enc2dec = nn.Linear(self.d_model, self.dec_dim)
        self.masking_ratio = configs.masking_ratio

        self.mamba_layers = nn.ModuleList([
            ResidualMambaBlock(
                mamba_class(
                    d_model=self.d_model,
                    d_state=configs.d_state,
                    d_conv=configs.d_conv,
                    expand=configs.expand,
                    layer_idx=i,
                    use_fast_path=True
                ),
                d_model=self.d_model
            ) for i in range(self.num_layers)
        ])

        # Mamba decoder
        self.pretrain_decoder = MambaDecoder(
            patch_size=self.patch_len,
            num_patches=self.seq_len // self.patch_len,
            d_model=self.d_model,
            d_decoder=self.dec_dim,
            num_layers=self.dec_depth,
            mamba_class=mamba_class,
            configs=configs)  # Mamba 초기화에 필요한 설정값 전달

        self.ar_norm = nn.LayerNorm(self.dec_dim)
        self.ar_pred = nn.Linear(self.dec_dim, self.enc_in)
        self.patch_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len//self.patch_len, self.d_model))
        # ==============================

        self.norm = RMSNorm(configs.d_model)

        self.stgat_layer = GAT_TCN(
            num_nodes = self.enc_in,
            in_channels=1,
            hidden_channels=128,
            out_channels=1,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            heads=3,
            num_layers=2,
            kernel_size=3
        )
        self.out_proj = nn.Linear(6, self.enc_in)

        nn.init.trunc_normal_(self.ar_token, std=.02)
        # nn.init.trunc_normal_(self.dec_pos_emb, std=.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.remask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def _norm_in(self, x):
        mean = x.mean(1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return (x - mean) / std, mean, std

    def pretrain_forward(self, x_enc):
        x_enc, mean, std = self._norm_in(x_enc)
        B, L, C = x_enc.shape
        num_patches = L // self.patch_len

        patches = x_enc.reshape(B, num_patches, self.patch_len, C).permute(0, 3, 1, 2)
        patches = patches.reshape(B * C, num_patches, self.patch_len)

        x_emb = self.patch_emb(patches)
        x_emb = x_emb + self.patch_pos_emb[:, :x_emb.size(1), :]

        mask_ratio = self.masking_ratio
        num_masked = max(1, int(mask_ratio * num_patches))

        # patch에 무작위로 부여되는 인덱스
        noise = torch.rand(x_emb.shape[0], num_patches, device=x_enc.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_mask = ids_shuffle[:, :num_masked]
        ids_visible = ids_shuffle[:, num_masked:]

        x_visible = torch.gather(x_emb, 1, ids_visible.unsqueeze(-1).expand(-1, -1, self.d_model))

        h = x_visible
        # construct the latent space based on only-visible patches
        for layer in self.mamba_layers:
            h = layer(h)
        latent_visible = self.norm(h)

        y_pred_masked = self.pretrain_decoder(latent_visible, ids_mask)
        y_true_masked = torch.gather(patches, 1, ids_mask.unsqueeze(-1).expand(-1, -1, self.patch_len))

        return y_pred_masked, y_true_masked

    def forecast(self, x_enc, x_mark_enc, training=True):
        x_norm, mean, std = self._norm_in(x_enc)
        x_enc_permuted = x_norm.permute(0, 2, 1) # [B, C, L]
        patches = x_enc_permuted.unfold(dimension=2, size=self.patch_len, step=self.stride)

        B, C, N, P = patches.shape
        patches = patches.reshape(B * C, N, P)
        x_mamba_emb = self.patch_emb(patches)

        for layer in self.mamba_layers:
            x_mamba_emb = layer(x_mamba_emb)
        x_mamba_emb = self.norm(x_mamba_emb)

        x_mamba = x_mamba_emb.reshape(B, C, N, self.d_model)
        x_mamba = self.d2p(x_mamba).reshape(B, C, N * self.patch_len) # [32, 7, 96]
        x_mamba = self.output_linear(x_mamba)  # [B, pred_len, C]
        x_mamba = x_mamba.permute(0, 2, 1)
        
        # 최종 출력 계산 (역정규화)
        x_out = x_mamba * std + mean

        return x_out
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        if self.task_name in ["short_term_forecast", "long_term_forecast"]:
            x_out = self.forecast(x_enc, x_mark_enc)
            # 최종 반환 형태에 맞게 슬라이싱
            return x_out
        return None
