import torch
import torch.nn.functional as F

def cal_corr_matrix(x: torch.Tensor, eps: float = 1e-8):
    """
    x: [B, C, L]
    returns: [B, C, C] Pearson correlation matrices
    """
    B, C, L = x.shape
    x_centered = x - x.mean(dim=-1, keepdim=True)              # [B, C, L]
    # 모수 공분산: sum / L
    cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (L)  # [B, C, C]
    # 모수 표준편차: var = sum / L  -> std(..., unbiased=False)가 동일
    std = x_centered.std(dim=-1, keepdim=True, unbiased=False)     # [B, C, 1]
    denom = torch.bmm(std, std.transpose(1, 2)) + eps               # [B, C, C]
    corr = cov / denom
    # 수치 안정 + 대각 1 보정 + 대칭화(작은 비대칭 잡음 제거)
    corr = torch.clamp(corr, -1.0, 1.0)
    eye = torch.eye(C, device=x.device).unsqueeze(0).expand(B, -1, -1)
    corr = corr * (1 - eye) + eye
    corr = 0.5 * (corr + corr.transpose(1, 2))
    return corr


def cal_spearman(x):
    """
    배치 내 각 샘플의 스피어만 상관관계 행렬을 계산합니다.
    입력 텐서 x의 형태는 [B, C, L] 이어야 합니다.
    """
    # 1. 시퀀스(L) 차원에 대해 각 값의 순위를 계산합니다.
    # argsort()를 두 번 적용하는 트릭을 사용해 순위를 효율적으로 계산할 수 있습니다.
    x_rank = x.argsort(dim=-1).argsort(dim=-1).float()

    # 2. 순위 텐서를 기존 피어슨 상관계수 계산 함수에 그대로 입력합니다.
    # (내부 로직은 동일)
    x_centered = x_rank - torch.mean(x_rank, dim=-1, keepdim=True)
    covariance = torch.bmm(x_centered, x_centered.transpose(1, 2))
    std_dev = torch.std(x_centered, dim=-1, keepdim=True)
    std_dev_matrix = torch.bmm(std_dev, std_dev.transpose(1, 2)) + 1e-5
    correlation_matrix = covariance / std_dev_matrix
    
    return correlation_matrix

def cal_cov_matrix(x):
    """
    Compute Covariance matrix for batched data.
    Input: x (Tensor): Batched time series data of shape [B, C, L]
                        (Batch, Channels, Length)
    Output: cov (Tensor): Batched covariance matrices of shape [B, C, C]
    """
    B, C, L = x.shape
    if L <= 1:
        # 샘플이 1개 이하면 공분산을 계산할 수 없습니다.
        return torch.zeros(B, C, C, device=x.device)
        
    # 1. L (시계열 길이)을 기준으로 평균을 계산합니다.
    # x_mean shape: [B, C, 1]
    x_mean = torch.mean(x, dim=2, keepdim=True)
    
    # 2. 데이터를 중앙화(center)합니다. (x - mean)
    # x_centered shape: [B, C, L]
    x_centered = x - x_mean
    
    # 3. 공분산 행렬 계산: (1 / (L - 1)) * (x_centered @ x_centered.T)
    # torch.bmm (Batch Matrix-Matrix product) 사용
    # (x_centered) @ (x_centered.transpose(1, 2))
    # [B, C, L] @ [B, L, C] -> [B, C, C]
    cov_matrix = (1.0 / (L - 1)) * torch.bmm(x_centered, x_centered.transpose(1, 2))
    
    return cov_matrix

def cal_sim(x: torch.Tensor, eps: float = 1e-8):
    """
    x: [B, C, L]
    returns: [B, C, C] Cosine similarity matrices
    """
    B, C, L = x.shape
    
    # 1. 분자: 채널 벡터 간의 내적(Dot Product)
    # [B, C, L] @ [B, L, C] -> [B, C, C]
    dot_product = torch.bmm(x, x.transpose(1, 2))
    
    # 2. 분모: 각 채널 벡터의 L2 Norm 계산
    # [B, C, L] -> [B, C, 1]
    norm_vec = torch.norm(x, p=2, dim=-1, keepdim=True)
    
    # 3. 분모 행렬: L2 Norm의 외적(Outer Product)
    # [B, C, 1] @ [B, 1, C] -> [B, C, C]
    denom = torch.bmm(norm_vec, norm_vec.transpose(1, 2)) + eps
    
    # 4. 코사인 유사도 계산
    sim = dot_product / denom
    
    # 5. 수치 안정성 및 대각 보정 (cal_corr_matrix와 동일)
    sim = torch.clamp(sim, -1.0, 1.0)
    eye = torch.eye(C, device=x.device).unsqueeze(0).expand(B, -1, -1)
    sim = sim * (1 - eye) + eye
    sim = 0.5 * (sim + sim.transpose(1, 2))
    
    return sim