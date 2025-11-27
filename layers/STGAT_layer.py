import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TemporalConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        padding = (0, kernel_size // 2)
        self.conv_1 = nn.Conv2d(in_channel, out_channel, (1, kernel_size), padding=padding)
        self.conv_2 = nn.Conv2d(in_channel, out_channel, (1, kernel_size), padding=padding)
        self.conv_3 = nn.Conv2d(in_channel, out_channel, (1, kernel_size), padding=padding)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        # X : [B, L, C, in_channel]
        X = X.permute(0, 3, 2, 1)  # [B, in_channel, C, L]
        P = self.conv_1(X) # 이미지처럼 [세로: 채널, 가로: 시간] 축으로 필터 적용
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = F.dropout(H, p=0.5, training=self.training)
        H = H.permute(0, 3, 2, 1) # [B, L, C, out_channel], out_channel = 128
        return H

class GAT(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # Residual Connection을 위한 초기 GAT 레이어
        '''GATConv
        입력 차원 : seq_len
        출력 차원 : hidden_channels
        heads : MHA head 수
        concat : 각 head의 출력을 concat해서 반환
        edge_dim : edge feature의 차원
        add_self_loops : 자기 자신과 연결되는 엣지를 추가해 자신의 이전 정보를 잃지 않도록 함
        fill_value : attention을 계산할 때, 연결되지 않은 엣지들을 무시하도록 만드는 masking 값
        '''
        # 처음에 seq_len 차원을 입력받아 선형변환으로 hidden_channels 차원으로 변환
        self.conv_0 = GATConv(seq_len, hidden_channels, heads=heads, concat=True, dropout=0.5, add_self_loops=True, edge_dim=1, fill_value=0, bias=True)
        self.bn_0 = nn.BatchNorm1d(hidden_channels * heads)

        # GAT 블록을 위한 레이어 리스트
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            # 첫 번째 레이어는 입력 차원이 seq_len, 나머지는 hidden_channels * heads
            in_dim = seq_len if i == 0 else hidden_channels * heads
            self.layers.append(
                GATConv(in_dim, hidden_channels, heads=heads, concat=True, dropout=0.5, add_self_loops=True, edge_dim=1)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.fc = nn.Linear(hidden_channels * heads, hidden_channels)
        self.activation = F.relu

    def forward(self, x, edge_index, edge_weight):

        # 입력 데이터를 한 개의 GAT 레이어만 통과시켜 얕은 특징을 추출
        x_0 = self.conv_0(x, edge_index, edge_weight)
        x_0 = self.bn_0(x_0)
        x_0 = self.activation(x_0)

        x_res = x
        # 여러 GAT layer를 사용해 노드 주변의 더 넓은 이웃 정보를 집계하고, 더 복잡한 공간적 패턴 학습
        for i in range(self.num_layers):
            # 이전 레이어의 출력(x_res)을 현재 레이어의 입력으로 사용
            x_res = self.layers[i](x_res, edge_index, edge_weight)
            x_res = self.bns[i](x_res)
            x_res = self.activation(x_res)
            x_res = F.dropout(x_res, 0.5, training=self.training)

        # Residual Connection과 최종 출력을 더함 
        x_out = x_res + x_0
        x_out = self.activation(x_out) # Residual 합산 후 활성화 함수 추가
        x_out = self.fc(x_out)
        x_out = F.dropout(x_out, 0.5, training=self.training)

        return x_out


class GAT_TCN(nn.Module):
    def __init__(self,
                 num_nodes: int,       # 데이터셋의 '전체' 변수 개수 (예: 7)
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 seq_len: int,
                 pred_len: int,
                 heads: int = 3,
                 num_layers: int = 2,
                 kernel_size: int = 3
    ):
        super(GAT_TCN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_channels = hidden_channels

        self.tcn_1 = TemporalConv(in_channels, hidden_channels, kernel_size=kernel_size)
        # tcn_1과 gat의 출력을 합친 결과를 다시 TCN에 통과
        self.tcn_2 = TemporalConv(hidden_channels * 2, hidden_channels, kernel_size=kernel_size)
        self.gat = GAT(num_nodes, seq_len, hidden_channels, hidden_channels, heads, num_layers)
        
        # 1. FC 레이어를 노드 단위 예측을 위해 재설계
        # 각 노드의 시계열 특징(seq_len * hidden_channels)을 예측 길이(pred_len)로 매핑
        self.pred_layer = nn.Linear(self.seq_len * self.hidden_channels, self.pred_len)

    def forward(self, x, edge_index, edge_weight) -> torch.FloatTensor:
        B, L, C = x.shape
        x_with_channel = x.unsqueeze(-1)    # 각 변수가 시점별로 channel feature를 갖도록 변환

        x_tcn = self.tcn_1(x_with_channel)
        x_gat_input = x.permute(0, 2, 1).contiguous().view(B * C, L)
        x_gat = self.gat(x_gat_input, edge_index, edge_weight)
        x_gat = x_gat.view(B, C, self.hidden_channels)
        x_gat = x_gat.unsqueeze(1).expand(-1, L, -1, -1) # [B, L, C, hidden_channels]
        fused_features = torch.cat([x_tcn, x_gat], dim=3) # [B, L, C, hidden_channels * 2]
        res = self.tcn_2(fused_features) # res shape: [B, L, N, hidden_channels]

        # 2. Reshape 방식 변경: 노드 차원을 유지하여 처리
        res = res.permute(0, 2, 1, 3) # [B, C, L, hidden_channels]
        res = res.reshape(B * C, self.seq_len * self.hidden_channels) # [B * C, L * hidden_channels]

        # 3. 노드별 독립적 예측
        res = self.pred_layer(res) # [B * C, pred_len]
        
        # 4. 최종 출력 형태로 복원
        res = res.view(B, C, self.pred_len)
        res = res.permute(0, 2, 1) # 최종 shape: [B, pred_len, C]

        return res
    
