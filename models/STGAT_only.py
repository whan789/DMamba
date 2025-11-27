import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from layers.RMSNorm import RMSNorm
from layers.STGAT_layer import GAT_TCN
from utils.cal_corr import cal_corr_matrix

class Model(nn.Module):
    def __init__(self, configs, mamba_class=None, dropout=0.1): # mamba_class는 호환성을 위해 남겨둠
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # STGAT Only이므로 Mamba, Patch 관련 파라미터 제거
        # Norm은 STGAT 입력 전 혹은 후에 사용할 수 있으므로 유지 (필요 시)
        # 여기서는 normalization을 _norm_in 메서드에서 직접 수행하므로 별도 LayerNorm이 필수적이지 않을 수 있으나
        # 코드 구조상 유지하거나 제거해도 무방함. 
        
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

    @torch.no_grad()
    def _norm_in(self, x):
        mean = x.mean(1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return (x - mean) / std, mean, std

    # STGAT Only 모델은 구조상 Masked Pretraining(Patch 복원)을 수행하기 어렵거나 
    # Mamba Encoder가 없으므로 해당 로직이 성립하지 않음.
    # 따라서 pretrain_forward는 제거하거나, 필요하다면 STGAT용으로 새로 짜야 함.
    # 여기서는 에러 방지를 위해 pass 처리 혹은 제거 추천.
    def pretrain_forward(self, x_enc):
        raise NotImplementedError("STGAT_only model does not support Mamba-based pretraining.")

    def forecast(self, x_enc, x_mark_enc, training=True):
        # 1. Normalization
        x_norm, mean, std = self._norm_in(x_enc) # [B, L, C]
        B, L, C = x_norm.shape
        
        # 2. Correlation Calculation을 위한 Permute
        x_enc_for_corr = x_norm.permute(0, 2, 1) # [B, C, L]
        x_enc_corr = cal_corr_matrix(x_enc_for_corr) # [B, C, C]

        # 3. Graph Construction (Batch processing)
        edge_indices = []
        edge_weights = []
        
        # GPU 연산 최적화를 위해 thresholding은 loop 밖에서 할 수도 있지만, 
        # sparse 변환이 그래프마다 다르므로 loop 유지
        for i in range(B):
            adj_matrix = x_enc_corr[i]
            threshold = 0.5
            adj_matrix[torch.abs(adj_matrix) < threshold] = 0
            edge_index, edge_weight = dense_to_sparse(adj_matrix)
            
            # Node index offset for batched graph
            edge_indices.append(edge_index + i * C)
            edge_weights.append(edge_weight)

        total_edge_index = torch.cat(edge_indices, dim=1)
        total_edge_weight = torch.cat(edge_weights, dim=0)

        # 4. STGAT Forward
        # x_norm: [B, L, C] -> STGAT가 내부적으로 처리 (보통 GAT_TCN 구현에 따라 다름)
        # 만약 GAT_TCN이 [B*C, L, 1] 형태를 원한다면 reshape이 필요할 수 있음.
        # 하지만 원본 Combined 모델에서도 x_mamba([B, L, C])를 넣었으므로 그대로 유지.
        x_graph_out = self.stgat_layer(
            x=x_norm,
            edge_index=total_edge_index,
            edge_weight=total_edge_weight
        )
        
        # 5. Denormalization
        x_out = x_graph_out * std + mean

        return x_out
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        if self.task_name in ["short_term_forecast", "long_term_forecast"]:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out
        return None