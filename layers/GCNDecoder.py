import torch.nn as nn
import torch

class GCNDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        # x shape: [B, L, C], adj shape: [B, C, C]
        # Adjacency matrix를 이용해 이웃 노드의 정보를 섞어줌
        x = torch.bmm(adj, x)   # [B, C, C] @ [B, C, L] -> [B, C, L]
        x = x.permute(0, 2, 1)  # [B, L, C]

        # 최종 복원을 위한 선형 변환
        x_2d = x.reshape(-1, x.size(-1))
        pred_2d = self.linear(x_2d)
        pred = pred_2d.view(x.size(0), x.size(1), -1)
        return pred