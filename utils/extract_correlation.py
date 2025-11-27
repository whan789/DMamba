import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.cal_corr import cal_spearman # 사용자 코드에 있는 함수 사용


def extract_correlations(model, test_loader, device):
    """
    Test Set 전체를 돌면서 Raw Data와 Mamba Embedding의 상관계수 행렬을 누적해서 평균을 구함
    """
    model.eval()
    
    total_raw_corr = None
    total_mamba_corr = None
    count = 0

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            # -------------------------------------------------------
            # 1. Raw Data 상관관계 계산
            # -------------------------------------------------------
            # Input: [B, L, N] -> [B, N, L] 형태로 변환 (N: 변수 개수, L: 시퀀스 길이)
            # cal_spearman은 보통 [Batch, Node, Time] 입력을 받음
            x_raw = batch_x.permute(0, 2, 1) 
            
            # Raw Data는 정규화(Norm) 후에 비교하는 것이 공정함 (Mamba도 Norm된 걸 쓰므로)
            x_raw_norm, _, _ = model._norm_in(batch_x)
            x_raw_norm = x_raw_norm.permute(0, 2, 1) # [B, N, L]
            
            raw_corr = cal_spearman(x_raw_norm) # [B, N, N]
            
            # -------------------------------------------------------
            # 2. Mamba Embedding 추출 (Model.forecast 내부 로직 재현)
            # -------------------------------------------------------
            # (1) Normalize & Patching
            x_norm, mean, std = model._norm_in(batch_x)
            x_enc_permuted = x_norm.permute(0, 2, 1) # [B, N, L]
            patches = x_enc_permuted.unfold(dimension=2, size=model.patch_len, step=model.stride)
            
            B, N, P_num, P_len = patches.shape
            # Model 코드에서 C가 Node 개수(enc_in)로 쓰이고 있음
            # reshape: [B * N, P_num, P_len]
            patches = patches.reshape(B * N, P_num, P_len) 
            
            # (2) Patch Embedding
            x_mamba_emb = model.patch_emb(patches) # [B*N, P_num, d_model]

            # (3) Mamba Encoder
            for layer in model.mamba_encoder:
                x_mamba_emb = layer(x_mamba_emb)
            x_mamba_emb = model.norm(x_mamba_emb)

            # (4) Representation 복원 (Projection)
            # [B, N, P_num, d_model]
            x_mamba = x_mamba_emb.reshape(B, N, P_num, model.d_model)
            # [B, N, P_num, P_len] -> [B, N, L] (Seq_len 복원)
            # d2p: d_model -> patch_len
            x_mamba = model.d2p(x_mamba).reshape(B, N, P_num * model.patch_len)
            
            mamba_corr = cal_spearman(x_mamba) # [B, N, N]

            # -------------------------------------------------------
            # 3. 평균 계산을 위한 누적
            # -------------------------------------------------------
            # 배치 내 평균
            batch_raw_mean = raw_corr.mean(dim=0).cpu().numpy()
            batch_mamba_mean = mamba_corr.mean(dim=0).cpu().numpy()

            if total_raw_corr is None:
                total_raw_corr = batch_raw_mean
                total_mamba_corr = batch_mamba_mean
            else:
                total_raw_corr += batch_raw_mean
                total_mamba_corr += batch_mamba_mean
            
            count += 1
            
            # (옵션) 너무 오래 걸리면 100 배치만 보고 break
            # if count >= 100: break

    # 전체 평균
    avg_raw_corr = total_raw_corr / count
    avg_mamba_corr = total_mamba_corr / count
    
    return avg_raw_corr, avg_mamba_corr