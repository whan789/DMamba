import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialToMe(nn.Module):
    def __init__(self, num_nodes, reduce_ratio=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        # 줄일 노드 개수 r 계산
        self.r = int(num_nodes * reduce_ratio)
        self.reduced_num_nodes = num_nodes - self.r

    def forward(self, x_feat, x_raw):
        """
        x_feat: [B, N, D] (유사도 계산용 - Mamba Output)
        x_raw:  [B, N, L] (실제 병합 대상 - GAT Input)
        """
        B, N, D = x_feat.shape
        L = x_raw.shape[2]
        
        # 1. Similarity Measure (Cosine)
        # 시간/채널 축 평균 -> 노드별 대표 벡터
        metric = x_feat.mean(dim=-1, keepdim=True) # [B, N, 1]
        metric = F.normalize(metric, p=2, dim=1)
        
        # Bipartite Partition (짝수/홀수)
        # A(Src): 짝수 인덱스, B(Dst): 홀수 인덱스
        # N이 홀수면 마지막 하나는 깍두기로 뺌
        protected = 0
        if N % 2 != 0:
            protected = 1
            
        n_src = (N - protected) // 2
        a = metric[:, 0:2*n_src:2, :] # [B, n_src, 1]
        b = metric[:, 1:2*n_src:2, :] # [B, n_src, 1]
        
        # 유사도 계산: A와 B 사이의 거리
        scores = a @ b.transpose(-1, -2) # [B, n_src, n_src]
        
        # 2. Matching (가장 유사한 쌍 찾기)
        # 각 A노드 입장에서 가장 친한 B노드 찾기
        node_max, node_idx = scores.max(dim=-1) # [B, n_src]
        
        # 상위 r개의 병합할 쌍(Edge) 선정
        # r이 n_src보다 클 수 없도록 클램핑
        k = min(self.r, n_src)
        top_k_scores, top_k_indices = torch.topk(node_max, k, dim=1) # [B, k]
        
        # 3. Create Merge Map (Unpooling을 위한 지도 만들기)
        # unpool_idx: [B, N] -> 각 원본 노드가 몇 번째 줄어든 노드로 가야 하는지 기록
        
        # 초기화: 일단 모든 노드는 자기 자신(혹은 순차적 인덱스)을 가리킴
        # 줄어든 후의 인덱스를 매기기 위한 준비
        
        # (구현의 효율성을 위해 'Scatter Mean' 방식을 사용합니다)
        # 병합될 A노드와 B노드에게 "너네는 이제 X번 노드야"라고 딱지를 붙여줍니다.
        
        device = x_feat.device
        
        # 그룹 ID를 부여할 텐서
        group_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).clone()
        
        # 병합 대상으로 선정된 top_k 쌍에 대해 ID 통일
        # top_k_indices는 A집합 기준 인덱스. 실제 원본 인덱스는 2*idx
        # 매칭된 B집합 인덱스는 node_idx가 가지고 있음. 원본 인덱스는 2*idx + 1
        
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        
        src_indices = top_k_indices * 2      # A 노드 원본 인덱스
        dst_indices = node_idx.gather(1, top_k_indices) * 2 + 1 # B 노드 원본 인덱스
        
        # A노드의 ID를 B노드 ID로 덮어씌움 (둘이 같은 그룹이 됨)
        # 실제로는 "더 작은 인덱스" 쪽으로 몰아주는 게 관례이나, 여기선 dst로 통일
        group_ids.scatter_(1, src_indices, group_ids.gather(1, dst_indices))
        
        # 이제 group_ids에는 구멍이 숭숭 뚫려있음 (합쳐져서 빈 번호 발생)
        # 이를 0 ~ (N-r-1)까지 차곡차곡 압축해야 함 (Relabeling)
        
        # unique를 쓰면 느리므로, 정렬 후 diff를 이용해 Relabeling
        sorted_ids, _ = group_ids.sort(dim=1)
        # 값이 바뀐 지점 찾기 (True/False)
        diff = torch.cat([torch.ones(B, 1, device=device, dtype=torch.bool), sorted_ids[:, 1:] != sorted_ids[:, :-1]], dim=1)
        # 누적합으로 0부터 순차적 ID 부여
        compact_ids_map = diff.cumsum(dim=1) - 1 # [B, N]
        
        # 원래 순서대로 compact_ids를 되돌려놔야 unpool_idx로 쓸 수 있음
        # 하지만 scatter_reduce를 하려면 그냥 sorted된 상태가 편함? -> No.
        # "원래 N개 노드가 어디로 가는지"를 알아야 함.
        
        # map_val: sorted_ids의 값이 원래 group_ids의 어디에 있었는지 역추적은 복잡함.
        # 더 빠른 방법: group_ids 값 자체를 랭크로 변환
        
        # (배치 루프 없이 하려면 복잡하므로, 여기선 안정적인 loop없는 trick 사용)
        # 하지만 unique가 가장 확실함. 속도를 위해 배치별 처리가 낫지만, 
        # 여기서는 Pytorch의 return_inverse 기능을 활용
        
        unpool_idx = torch.empty(B, N, dtype=torch.long, device=device)
        for i in range(B):
            _, inverse = torch.unique(group_ids[i], return_inverse=True)
            unpool_idx[i] = inverse
            
        # 4. Merge Execution (Mean Pooling)
        # unpool_idx를 이용해 x_raw 데이터를 합침
        # x_raw: [B, N, L] -> [B, N_new, L]
        
        N_new = unpool_idx.max() + 1
        
        # 초기화
        x_raw_small = torch.zeros(B, N_new, L, device=device)
        x_feat_small = torch.zeros(B, N_new, D, device=device)
        
        # 개수 세기 (평균 내려고)
        counts = torch.zeros(B, N_new, 1, device=device)
        ones = torch.ones(B, N, 1, device=device)
        
        # Scatter Add
        # index shape 맞춰주기: [B, N, L]
        idx_exp_L = unpool_idx.unsqueeze(-1).expand(-1, -1, L)
        idx_exp_D = unpool_idx.unsqueeze(-1).expand(-1, -1, D)
        idx_exp_1 = unpool_idx.unsqueeze(-1)
        
        x_raw_small.scatter_add_(1, idx_exp_L, x_raw)
        x_feat_small.scatter_add_(1, idx_exp_D, x_feat)
        counts.scatter_add_(1, idx_exp_1, ones)
        
        # 평균 계산
        x_raw_small = x_raw_small / (counts + 1e-6)
        x_feat_small = x_feat_small / (counts + 1e-6)
        
        # N_new가 배치마다 다를 수 있음 (top-k가 같아도 unique 개수는 다를 수 있음 희박하지만)
        # 하지만 위 로직상 top-k가 고정이라 N_new는 고정됨 (N-r)
        
        return x_feat_small, x_raw_small, unpool_idx

    def unpool(self, x_out, unpool_idx):
        """
        논문 방식: Cloning (Gather를 이용한 복제)
        x_out: [B, N_new, Pred_Len] - GAT 통과 후
        unpool_idx: [B, N] - 원본 노드가 가리키는 Reduced 인덱스
        """
        B, N_new, P = x_out.shape
        B, N = unpool_idx.shape
        
        # Gather를 위해 인덱스 확장
        # unpool_idx: [B, N] -> [B, N, P]
        idx_exp = unpool_idx.unsqueeze(-1).expand(-1, -1, P)
        
        # x_out에서 값을 가져옴 (복제됨)
        x_restored = torch.gather(x_out, 1, idx_exp) # [B, N, P]
        
        return x_restored