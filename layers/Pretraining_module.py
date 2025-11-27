import torch
import torch.nn as nn
from layers.Residual_Mamba import ResidualMambaBlock
from layers.RMSNorm import RMSNorm

class MambaDecoder(nn.Module):
    def __init__(self, patch_size, num_patches, d_model, num_layers, mamba_class, configs):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.patch_size = patch_size

        # 마스크 토큰과 위치 임베딩은 기존과 동일
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.d_model))

        # === 핵심 변경: Transformer 블록을 Mamba 블록으로 교체 ===
        self.mamba_layers = nn.ModuleList([
            ResidualMambaBlock(
                mamba_class(
                    d_model=self.d_model,
                    d_state=configs.d_state,
                    d_conv=configs.d_conv,
                    expand=configs.expand,
                ),
                d_model=self.d_model
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(self.d_model)
        # =======================================================

        # 최종 예측을 위한 Head는 기존과 동일
        self.head = nn.Linear(self.d_model, patch_size)

        # 가중치 초기화
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def forward(self, x_visible, ids_visible):
        B, N_vis, D = x_visible.shape

        # 1. 전체 시퀀스 재구성
        # 모든 위치에 마스크 토큰을 채운 텐서 생성
        x_full = self.mask_token.expand(B, self.num_patches, -1).clone()

        x_full.scatter_(
            dim=1, 
            index=ids_visible.unsqueeze(-1).expand(-1, -1, D), 
            src=x_visible
        )

        # 2. 위치 임베딩 추가
        x = x_full + self.pos_embed[:, :self.num_patches, :]

        for layer in self.mamba_layers:
            x = layer(x)
        x = self.norm(x)

        # 5. 전체 시퀀스에 대한 예측값 반환
        # shape: [B, num_patches, patch_size]
        y_pred = self.head(x)
        
        return y_pred

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

class TransformerDecoder(nn.Module):
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

        pos_embed = self.pos_embed.expand(B, -1, -1).clone()
        decoder_tokens = self.mask_token.expand(B, self.num_patches, -1).clone()

        bool_mask_vis = torch.zeros(B, self.num_patches, dtype=torch.bool, device=x_visible.device)
        ids_vis = torch.arange(self.num_patches, device=x_visible.device).unsqueeze(0).expand(B, -1)
        bool_mask_vis.scatter_(1, ids_mask, True)
        bool_mask_vis = ~bool_mask_vis

        decoder_tokens[bool_mask_vis] = x_visible.reshape(-1, D)

        x = decoder_tokens + pos_embed

        for blk in self.blocks:
            x = blk(q=x, kv=x_visible)

        x_masked = x[~bool_mask_vis].reshape(B, -1, self.d_decoder)

        y_pred = self.head(x_masked)
        return y_pred

# CrossAttention Decoder
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

# Transformer Decoder
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv):
        q = q + self.drop_path(self.attn(self.norm1_q(q), self.norm1_kv(kv)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q