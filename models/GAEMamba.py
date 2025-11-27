import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.gtnet import gtnet
from layers.Residual_Mamba import ResidualMambaBlock
from layers.RMSNorm import RMSNorm
from layers.RevIN import RevIN
from layers.GCNDecoder import GCNDecoder

# layers/gtnet.py 또는 models/your_model.py 상단에 추가

    
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

        # Mamba Layers
        self.layers = nn.ModuleList([
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
        self.input_proj = nn.Linear(configs.residual_channels * self.enc_in, self.d_model)
        self.fuse_gate       = nn.Linear(self.d_model, self.d_model)

        self.channel_proj = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.len_proj = nn.Linear(78, self.pred_len)

        # Graph block + gate fusion
        self.graph_layer = gtnet(configs=configs,
                                predefined_A=getattr(configs, "predefined_A", None),
                                static_feat=getattr(configs, "static_feat", None))
        
        self.gnn_decoder = GCNDecoder(self.enc_in, self.enc_in)
        self.encoder_mask_token = nn.Parameter(torch.zeros(1, self.enc_in))
        self.decoder_mask_token = nn.Parameter(torch.zeros(1, self.enc_in))
                                
        self.loss_lambda = configs.loss_lambda
        self.loss_alpha = configs.loss_alpha
        self.loss_beta = configs.loss_beta

    @torch.no_grad()
    def _norm_in(self, x):
        mean = x.mean(1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return (x - mean) / std, mean, std

    def _graph_encode(self, x_norm):
        # x_norm: [B, L, C]
        x_graph_input = x_norm.permute(0, 2, 1).unsqueeze(1)  # [B, 1, C, L]
        x_graph_out, _, dense_adp = self.graph_layer(x_graph_input)   # [B, C, L, 1]
        x_graph_out = x_graph_out.squeeze(-1).permute(0, 2, 1)  # [B, L, C]
        return x_graph_out, dense_adp

    def pretrain_forward(self, x_enc, mask_ratio=0.5):
        B, L, C = x_enc.shape

        # 1) 정규화
        x_norm, mean, std = self._norm_in(x_enc)

        # 2) 채널 마스크 샘플링
        k = max(1, int(mask_ratio * C))
        mask_idx = torch.zeros(B, C, device=x_enc.device, dtype=torch.bool) # 어디를 가릴지 위치를 설정

        # 배치 내의 각 데이터 샘플마다 서로 다른 채널을 랜덤하게 마스킹하기 위한 마스크를 생성하는 부분
        for b in range(B):
            perm = torch.randperm(C, device=x_enc.device)
            mask_idx[b, perm[:k]] = True

        # 3) 입력 마스킹 (마스킹된 위치를 학습 가능한 토큰으로 교체)
        x_masked = x_norm.clone()
        broadcast_mask = mask_idx.unsqueeze(1).expand(-1, L, -1) # [B, L, C]
        x_masked[broadcast_mask] = self.encoder_mask_token.expand(B, L, C)[broadcast_mask]

        # 4) 그래프 인코더 통과 (인접 행렬도 받음)
        z, dense_adp = self._graph_encode(x_masked) 
        dense_adp = dense_adp.unsqueeze(0).expand(B, -1, -1)    # [B, C, C]

        z_remasked = z.clone()
        z_remasked = z_remasked.permute(0, 2, 1)
        z_remasked[broadcast_mask] = self.decoder_mask_token.expand(B, L, C)[broadcast_mask]
        z_remasked = z_remasked.permute(0, 2, 1)

        # 6) GNN 디코더로 채널 복원
        pred_norm = self.gnn_decoder(z_remasked, dense_adp)

        # 7) 손실 계산 (마스킹된 채널에서만)

        # 손실 계산을 위해 텐서 모양을 [B*L, C]로 변환
        x_norm = x_norm.permute(0,2,1)
        pred_norm = pred_norm.permute(0,2,1)

        # 마스킹된 위치의 특징만 추출
        masked_true = x_norm[mask_idx]
        masked_pred = pred_norm[mask_idx]

        loss = sce_loss(masked_pred, masked_true, alpha=2)
        return loss

    def forecast(self, x_enc, x_mark_enc):
        x_norm, mean, std = self._norm_in(x_enc)

        # Step 1. Graph 경로
        x_graph_input = x_norm.permute(0, 2, 1).unsqueeze(1)
        x_graph_out, _, _ = self.graph_layer(x_graph_input) # [B, residual channel, enc_in, seq_len]
        B, C, N, L = x_graph_out.shape
        
        x_graph_out = x_graph_out.permute(0,3,1,2).reshape(B, L, C * N)
        x_projected = self.input_proj(x_graph_out)

        x_mark_adjusted = x_mark_enc[:, -L:, :]
        x_mamba_input = self.embedding(x_projected, x_mark_adjusted)
        print('mamba input shape : ', x_mamba_input.shape)

        x_mamba = self.norm(x_mamba_input)
       
        for layer in self.layers:
            x_mamba = layer(x_mamba)
            x_mamba = self.norm(x_mamba)
            
        x_out = x_mamba.permute(0, 2, 1) 
        x_out = self.len_proj(x_out)
        x_out = x_out.permute(0, 2, 1)
        x_out = self.channel_proj(x_out)
        x_out = x_out * std + mean
        return x_out

    # <<< MODIFIED START >>>
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        본학습(Fine-tuning) 및 테스트를 위한 forward 함수.
        이제 모델은 예측값만 반환합니다. 손실 계산은 외부(Exp 클래스)에서 수행합니다.
        """
        if self.task_name in ["short_term_forecast", "long_term_forecast"]:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
        return None
    