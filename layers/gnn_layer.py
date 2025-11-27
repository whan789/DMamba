import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_Aggregator import MambaAggregator

class MambaGNNLayer(nn.Module):
    """
    Vectorized MambaGNNLayer using padding to handle variable number of neighbors.
    This version avoids slow Python loops for much faster execution on GPU.
    """
    def __init__(self, configs, mamba_class, dropout=0.1):
        super().__init__()
        self.mamba_aggregator = MambaAggregator(configs, mamba_class)
        d_model = configs.d_model
        self.update_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, adj_matrix):
        # node_features: (batch_size, num_nodes, d_model)
        # adj_matrix: (num_nodes, num_nodes)
        batch_size, num_nodes, d_model = node_features.shape
        device = node_features.device

        # --- Vectorized Neighbor Finding (Python 루프 제거) ---
        # 1. 0이 아닌 값(엣지)의 인덱스를 찾습니다. (row, col) 형태로 나옵니다.
        #    row는 중심 노드, col은 이웃 노드를 의미합니다.
        adj_matrix_sparse = adj_matrix.to_sparse()
        src_nodes, neighbor_nodes = adj_matrix_sparse.indices()
        
        # 엣지가 없는 경우를 위한 예외 처리
        if src_nodes.numel() == 0:
            agg_features = torch.zeros_like(node_features)
            cons_loss = torch.tensor(0.0, device=device)
        else:
            # 2. 각 노드별 이웃의 수를 계산합니다.
            num_neighbors = torch.bincount(src_nodes, minlength=num_nodes)
            max_neighbors = num_neighbors.max().item()

            # 3. 패딩된 이웃 인덱스 텐서와 마스크를 생성합니다.
            padded_indices = torch.full((num_nodes, max_neighbors), fill_value=0, dtype=torch.long, device=device)
            mask = torch.zeros((num_nodes, max_neighbors), dtype=torch.bool, device=device)

            # 4. 각 노드의 이웃 리스트 내에서 이웃의 순서 인덱스를 계산합니다.
            #    (예: 노드 A의 첫번째 이웃, 두번째 이웃...)
            cumsum_neighbors = torch.cat([torch.tensor([0], device=device), torch.cumsum(num_neighbors, 0)[:-1]])
            within_node_indices = torch.arange(len(src_nodes), device=device) - cumsum_neighbors[src_nodes]

            # 5. 계산된 인덱스를 사용해 padded_indices 텐서를 한 번에 채웁니다.
            padded_indices[src_nodes, within_node_indices] = neighbor_nodes
            mask[src_nodes, within_node_indices] = True
            
            # --- 이하는 기존 로직과 거의 동일 ---
            neighbor_features_padded = node_features[:, padded_indices, :]
            
            mask_expanded = mask.unsqueeze(0).unsqueeze(-1) # 브로드캐스팅을 위해 차원 추가
            neighbor_features_padded = neighbor_features_padded * mask_expanded.float()

            mamba_input = neighbor_features_padded.reshape(batch_size * num_nodes, max_neighbors, d_model)
            mamba_mask = mask.expand(batch_size, -1, -1).reshape(batch_size * num_nodes, max_neighbors)
            
            agg_info, cons_loss_per_node = self.mamba_aggregator.forward_batch(mamba_input, mamba_mask)
            
            agg_features = agg_info.view(batch_size, num_nodes, d_model)
            cons_loss = cons_loss_per_node.mean()

        # 노드 특징 업데이트
        combined_info = torch.cat([node_features, agg_features], dim=-1)
        update_info = self.update_mlp(combined_info)
        
        out_features = self.norm(node_features + self.dropout(update_info))
        
        return out_features, cons_loss