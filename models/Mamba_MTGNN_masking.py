
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.gtnet import gtnet
from layers.Residual_Mamba import ResidualMambaBlock
from layers.RMSNorm import RMSNorm
from utils.cal_corr import cal_corr_matrix

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class PretrainDecoder(nn.Module):
    def __init__(self, patch_size, num_patches, d_model, d_decoder, num_heads, num_layers):
        super().__init__()
        self.d_decoder = d_decoder
        self.num_patches = num_patches

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_decoder))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_decoder))
        
        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=d_decoder,
                num_heads=num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            ) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(d_decoder)
        self.head = nn.Linear(d_decoder, patch_size)

        # 가중치 초기화
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def forward(self, x_visible, ids_mask):
        B, N_vis, D = x_visible.shape
        
        # 전체 패치 수에 맞는 디코더 토큰 준비 (마스크 토큰 + 위치 임베딩)
        # clone()을 사용해 pos_embed 원본이 변경되지 않도록 함
        pos_embed = self.pos_embed.expand(B, -1, -1).clone()
        decoder_tokens = self.mask_token.expand(B, self.num_patches, -1).clone()
        
        # 마스킹되지 않은 위치에는 인코더 출력을, 마스킹된 위치에는 마스크 토큰을 배치
        # 1. bool 마스크 생성
        bool_mask_vis = torch.zeros(B, self.num_patches, dtype=torch.bool, device=x_visible.device)
        ids_vis = torch.arange(self.num_patches, device=x_visible.device).unsqueeze(0).expand(B, -1)
        bool_mask_vis.scatter_(1, ids_mask, True) # 마스킹된 곳이 True
        bool_mask_vis = ~bool_mask_vis # 마스킹되지 않은 곳이 True

        # 2. 값 채워넣기
        decoder_tokens[bool_mask_vis] = x_visible.reshape(-1, D)

        # 위치 임베딩 더하기
        x = decoder_tokens + pos_embed

        # Cross-Attention Decoder 실행
        # 마스킹되지 않은 토큰(x_visible)을 key와 value로 사용하고,
        # 전체 토큰(x)을 query로 사용하여 마스킹된 부분을 복원
        for blk in self.blocks:
            x = blk(q=x, kv=x_visible) # q는 전체 토큰, kv는 컨텍스트(마스킹 안된 토큰)

        # 마스킹된 부분만 예측
        x_masked = x[~bool_mask_vis].reshape(B, -1, self.d_decoder)
        
        y_pred = self.head(x_masked)
        return y_pred

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        B, N, C = q.shape
        # kv B, N, C
        kv_B, kv_N, kv_C = kv.shape
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(kv).reshape(kv_B, kv_N, 2, self.num_heads, kv_C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() # DropPath 생략
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv):
        q = q + self.drop_path(self.attn(self.norm1_q(q), self.norm1_kv(kv)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q

    
class Model(nn.Module):
    def __init__(self, configs, mamba_class, dropout=0.1):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in  # number of variables
        self.embed = configs.embed
        self.freq = configs.freq
        self.dropout = dropout
        self.patch_len = configs.patch_len

        # Embedding
        # gtnet이 추출한 d_model 차원의 특징과 시간 특징을 결합
        self.embedding = DataEmbedding(self.d_model, self.d_model, configs.embed, configs.freq, self.dropout)
        self.mamba_linear = nn.Linear(self.enc_in, self.d_model)
        self.d2p = nn.Linear(self.d_model, self.patch_len)
        self.output_linear = nn.Linear(self.d_model, self.enc_in)
        self.stride = getattr(configs, 'stride', 8)
        
        # 패치 임베딩을 위한 Linear 레이어
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        
        # 다음 패치 예측을 위한 헤드
        self.decoder = nn.Sequential(nn.Linear(self.d_model, self.d_model * 2),
                                     nn.GELU(),
                                     nn.Linear(self.d_model * 2, self.patch_len))

        # Mamba Layers
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
            ) for i in range(configs.e_layers)
        ])
        self.norm = RMSNorm(configs.d_model)

        # Graph block + gate fusion
        self.graph_layer = gtnet(configs=configs,
                                predefined_A=getattr(configs, "predefined_A", None),
                                static_feat=getattr(configs, "static_feat", None))
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        # 디코더 입력(인코더 출력) re-masking을 위한 learnable token
        self.remask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        
        self.dec_dim = configs.d_model
        self.dec_depth = 4
        self.dec_nhead = 8
        self.ar_token = nn.Parameter(torch.zeros(1,1, self.dec_dim))
        self.dec_pos_emb = nn.Parameter(torch.zeros(1, self.pred_len, self.dec_dim))
        self.enc2dec = nn.Linear(self.d_model, self.dec_dim)

         # Cross-Attention 기반의 디코더 블록
        self.dec_block = nn.ModuleList([
            DecoderBlock(
                dim=self.dec_dim,
                num_heads=self.dec_nhead,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            ) for _ in range(self.dec_depth)
        ])

        self.pretrain_decoder = PretrainDecoder(
            patch_size=self.patch_len,
            num_patches=self.seq_len // self.patch_len,
            d_model=self.d_model,
            d_decoder=self.dec_dim, # configs.d_model과 동일하게 설정 가능
            num_heads=self.dec_nhead,
            num_layers=self.dec_depth
        )

        # 디코더 출력 정규화 및 최종 예측을 위한 레이어
        self.ar_norm = nn.LayerNorm(self.dec_dim)
        # 디코더의 출력(decoder_embed_dim)을 원래 변수 개수(enc_in)로 변환
        self.ar_pred = nn.Linear(self.dec_dim, self.enc_in) 
        self.patch_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len//self.patch_len, self.d_model))

        # 가중치 초기화
        nn.init.trunc_normal_(self.ar_token, std=.02)
        nn.init.trunc_normal_(self.dec_pos_emb, std=.02)
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
        original_timeseries = x_enc
        x_enc, mean, std = self._norm_in(x_enc)
        B, L, C = x_enc.shape
        num_patches = L // self.patch_len
        
        patches = x_enc.reshape(B, num_patches, self.patch_len, C).permute(0, 3, 1, 2)
        original_patches = patches.reshape(B * C, num_patches, self.patch_len)
        
        x_emb = self.patch_embedding(original_patches)
        x_emb = x_emb + self.patch_pos_emb[:, :x_emb.size(1), :]

        mask_ratio = 0.5
        num_masked = max(1, int(mask_ratio * num_patches))
        
        # 마스킹할 인덱스와 마스킹되지 않을 인덱스를 분리
        noise = torch.rand(x_emb.shape[0], num_patches, device=x_enc.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_mask = ids_shuffle[:, :num_masked]
        ids_visible = ids_shuffle[:, num_masked:]
        
        # 마스킹되지 않은 토큰만 인코더에 입력
        x_visible = torch.gather(x_emb, 1, ids_visible.unsqueeze(-1).expand(-1, -1, self.d_model))
        
        h = x_visible
        for layer in self.mamba_layers:
            h = layer(h)
        latent_visible = self.norm(h)
        
        # 디코더로 마스킹된 패치 예측
        # 인코더 출력(latent_visible)과 마스킹된 위치 정보(ids_mask)를 전달
        y_pred_masked = self.pretrain_decoder(latent_visible, ids_mask)
        
        # 실제 마스킹된 패치 추출
        y_true_masked = torch.gather(original_patches, 1, ids_mask.unsqueeze(-1).expand(-1, -1, self.patch_len))

        original_visible_patches = torch.gather(original_patches, 1, ids_visible.unsqueeze(-1).expand(-1, -1, self.patch_len))
        reconstructed_patches = torch.zeros_like(original_patches)
        reconstructed_patches.scatter_(1, ids_visible.unsqueeze(-1).expand(-1, -1, self.patch_len), original_visible_patches)
        reconstructed_patches.scatter_(1, ids_mask.unsqueeze(-1).expand(-1, -1, self.patch_len), y_pred_masked)

        # 1. (B * C, num_patches, patch_len) -> (B, C, num_patches, patch_len)
        reconstructed_patches_reshaped = reconstructed_patches.reshape(B, C, num_patches, self.patch_len)
        
        # 2. (B, C, num_patches, patch_len) -> (B, C, L)
        reconstructed_timeseries_reshaped = reconstructed_patches_reshaped.reshape(B, C, L)
        
        # 3. (B, C, L) -> (B, L, C) : 원본 시계열 형태로 최종 복원
        reconstructed_timeseries = reconstructed_timeseries_reshaped.permute(0, 2, 1)

        # 재정규화 (선택 사항): 복원된 값에 원래 mean, std를 다시 적용
        reconstructed_timeseries = reconstructed_timeseries * (std + 1e-6) + mean

        corr_origin = cal_corr_matrix(original_timeseries)
        corr_recon = cal_corr_matrix(reconstructed_timeseries)

        return y_pred_masked, y_true_masked, corr_origin, corr_recon

    def forecast(self, x_enc, x_mark_enc):
        x_norm, mean, std = self._norm_in(x_enc)
        x_enc = x_norm.permute(0, 2, 1) # [B, C, L]
        patches = x_enc.unfold(dimension=2, size=self.patch_len, step=self.stride)

        B, C, N, P = patches.shape
        patches = patches.reshape(B * C, N, P)
        x_mamba = self.patch_embedding(patches) # [B*V, N-1, d_model]

        for layer in self.mamba_layers:
            x_mamba = layer(x_mamba)
        x_mamba = self.norm(x_mamba)
        
        x_mamba = x_mamba.reshape(B, C, N, self.d_model)
        x_mamba = self.d2p(x_mamba).reshape(B, C, N * self.patch_len)

        x_enc_corr = cal_corr_matrix(x_enc)
        x_mamba_corr = cal_corr_matrix(x_mamba)

        x_graph_input = x_mamba.unsqueeze(1) # 32, 1, 512, 96
        x_graph_out, _, _ = self.graph_layer(x_graph_input) # 32, 96, 7, 1

        x_graph_out = x_graph_out.squeeze(-1)
        x_out = x_graph_out * std + mean

        return x_out, x_enc_corr, x_mamba_corr

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        if self.task_name in ["short_term_forecast", "long_term_forecast"]:
            x_out, x_enc_corr, x_mamba_corr = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :], x_enc_corr, x_mamba_corr
        return None