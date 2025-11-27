
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.gtnet import gtnet
from layers.Residual_Mamba import ResidualMambaBlock
from layers.RMSNorm import RMSNorm

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

        # Embedding
        # gtnet이 추출한 d_model 차원의 특징과 시간 특징을 결합
        self.embedding = DataEmbedding(self.d_model, self.d_model, configs.embed, configs.freq, self.dropout)
        self.mamba_linear = nn.Linear(self.enc_in, self.d_model)
        self.graph_linear = nn.Linear(self.d_model, self.enc_in)
        self.output_linear = nn.Linear(self.d_model, self.enc_in)

        self.patch_len = getattr(configs, 'patch_len', 8)
        self.stride = getattr(configs, 'stride', 8)
        
        # 패치 임베딩을 위한 Linear 레이어
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        
        # 다음 패치 예측을 위한 헤드
        self.pretrain_head = nn.Linear(self.d_model, self.patch_len)

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

        # 디코더 출력 정규화 및 최종 예측을 위한 레이어
        self.ar_norm = nn.LayerNorm(self.dec_dim)
        # 디코더의 출력(decoder_embed_dim)을 원래 변수 개수(enc_in)로 변환
        self.ar_pred = nn.Linear(self.dec_dim, self.enc_in) 

        # 가중치 초기화
        nn.init.trunc_normal_(self.ar_token, std=.02)
        nn.init.trunc_normal_(self.dec_pos_emb, std=.02)
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
        """
        Overlapping Autoregressive Pre-training을 위한 forward 함수
        Args:
            x_enc: [Batch, Sequence Length, Variables]
                   데이터 로더에서 전달된 하나의 긴 시계열 조각.
                   Sequence Length는 패치화가 가능한 충분한 길이어야 함 (예: 512)
        """
        # 입력 데이터 정규화
        x_enc, mean, std = self._norm_in(x_enc)
        
        # 1. Overlapping Patching
        # x_enc: [B, L, C] -> patches: [B, N, C, P]
        # B: Batch, L: Sequence Length, C: Variables, N: Num Patches, P: Patch Length
        x_enc = x_enc.permute(0, 2, 1)  # [B, C, L]
        patches = x_enc.unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 2, 1, 3) # [B, N, C, P]

        # 변수(Variable) 차원을 배치 차원으로 합침 (Mamba는 3D 텐서 입력을 가정)
        B, N, C, P = patches.shape
        patches = patches.reshape(B * C, N, P)

        # 2. 자기회귀(Autoregressive)를 위한 입력(x)과 정답(y) 생성
        # 이전 패치들을 보고 다음 패치를 예측
        x_pretrain = patches[:, :-1, :]  # 입력: [B*V, N-1, P]
        y_pretrain = patches[:, 1:, :]   # 정답: [B*V, N-1, P]

        # 3. 패치 임베딩
        # 각 패치를 d_model 차원의 벡터로 변환
        x_emb = self.patch_embedding(x_pretrain) # [B*V, N-1, d_model]
        
        # 4. Mamba 인코더 통과
        for layer in self.mamba_layers:
            x_emb = layer(x_emb)
        x_emb = self.norm(x_emb)

        # 5. 다음 패치 예측
        y_pred = self.pretrain_head(x_emb) # [B*V, N-1, P]
        
        # 6. 정답(y) 반환 (손실 계산을 위해)
        # y_pred와 y_pretrain의 차원을 맞춰주었으므로, y_pretrain을 그대로 반환
        return y_pred, y_pretrain

    def forecast(self, x_enc, x_mark_enc):
        x_norm, mean, std = self._norm_in(x_enc)
        x_norm = self.mamba_linear(x_norm)
        x_mamba = self.norm(x_norm)

        for layer in self.mamba_layers:
            x_mamba = layer(x_mamba)
            x_mamba = self.norm(x_mamba)

        x_mamba = self.graph_linear(x_mamba)

        # Step 1. Graph 경로
        x_graph_input = x_mamba.unsqueeze(1) # 32, 1, 512, 96
        x_graph_input = x_graph_input.permute(0,1,3,2)
        x_graph_out, _, _ = self.graph_layer(x_graph_input) # 32, 96, 7, 1

        x_graph_out = x_graph_out.squeeze(-1)
        x_out = x_graph_out * std + mean

        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        if self.task_name in ["short_term_forecast", "long_term_forecast"]:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
        return None