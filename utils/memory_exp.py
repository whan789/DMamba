import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse # __main__에 필요
from collections import defaultdict # 프로파일러에 필요
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# --- 필요한 모듈 임포트 (실제 경로에 맞게 수정 필요) ---
# 아래는 예시 경로이므로, 실제 프로젝트 구조에 맞게 수정하세요.
from layers.Residual_Mamba import ResidualMambaBlock
from layers.RMSNorm import RMSNorm
from layers.STGAT_layer import GAT_TCN
from layers.Pretraining_module import TransformerDecoder, DecoderBlock, MambaDecoder
from layers.VarDrop import efficient_sampler
from utils.cal_corr import cal_corr_matrix
from mamba_ssm import Mamba
# ----------------------------------------------------

class Model(nn.Module):
    def __init__(self, configs, mamba_class, dropout=0.1):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.embed = configs.embed
        self.freq = configs.freq
        self.dropout = dropout
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.use_vardrop = configs.use_vardrop
        self.k = configs.vardrop_k
        self.group_size = configs.group_size
        self.num_layers = configs.e_layers
        rfft_len = (self.seq_len) // 2 + 1
        lpf_cutoff = min(25, rfft_len // 2) 
        self.freq_list = list(range(lpf_cutoff))

        self.mamba_linear = nn.Linear(self.enc_in, self.d_model)
        self.d2p = nn.Linear(self.d_model, self.patch_len)
        self.output_linear = nn.Linear(self.d_model, self.enc_in)
        self.patch_emb = nn.Linear(self.patch_len, self.d_model)

        # ====== Pretraining Args ======
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.remask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.dec_dim = self.d_model
        self.dec_depth = 2
        self.dec_nhead = 8
        self.ar_token = nn.Parameter(torch.zeros(1,1, self.dec_dim))
        # self.dec_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.dec_dim))
        self.enc2dec = nn.Linear(self.d_model, self.dec_dim)
        self.masking_ratio = configs.masking_ratio

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
            ) for i in range(self.num_layers)
        ])

        # Mamba decoder
        self.pretrain_decoder = MambaDecoder(
            patch_size=self.patch_len,
            num_patches=self.seq_len // self.patch_len,
            d_model=self.d_model,
            d_decoder=self.dec_dim,
            num_layers=self.dec_depth,
            mamba_class=mamba_class,
            configs=configs)  # Mamba 초기화에 필요한 설정값 전달

        self.ar_norm = nn.LayerNorm(self.dec_dim)
        self.ar_pred = nn.Linear(self.dec_dim, self.enc_in)
        self.patch_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len//self.patch_len, self.d_model))
        # ==============================

        self.norm = RMSNorm(configs.d_model)

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
        self.out_proj = nn.Linear(6, self.enc_in)

        nn.init.trunc_normal_(self.ar_token, std=.02)
        # nn.init.trunc_normal_(self.dec_pos_emb, std=.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.remask_token, std=.02)
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
        x_enc, mean, std = self._norm_in(x_enc)
        B, L, C = x_enc.shape
        num_patches = L // self.patch_len

        patches = x_enc.reshape(B, num_patches, self.patch_len, C).permute(0, 3, 1, 2)
        patches = patches.reshape(B * C, num_patches, self.patch_len)

        x_emb = self.patch_emb(patches)
        x_emb = x_emb + self.patch_pos_emb[:, :x_emb.size(1), :]

        mask_ratio = self.masking_ratio
        num_masked = max(1, int(mask_ratio * num_patches))

        # patch에 무작위로 부여되는 인덱스
        noise = torch.rand(x_emb.shape[0], num_patches, device=x_enc.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_mask = ids_shuffle[:, :num_masked]
        ids_visible = ids_shuffle[:, num_masked:]

        x_visible = torch.gather(x_emb, 1, ids_visible.unsqueeze(-1).expand(-1, -1, self.d_model))

        h = x_visible
        # construct the latent space based on only-visible patches
        for layer in self.mamba_layers:
            h = layer(h)
        latent_visible = self.norm(h)

        y_pred_masked = self.pretrain_decoder(latent_visible, ids_mask)
        y_true_masked = torch.gather(patches, 1, ids_mask.unsqueeze(-1).expand(-1, -1, self.patch_len))

        return y_pred_masked, y_true_masked

    def forecast(self, x_enc, x_mark_enc, training=True):
        x_norm, mean, std = self._norm_in(x_enc)
        x_enc_permuted = x_norm.permute(0, 2, 1) # [B, C, L]
        patches = x_enc_permuted.unfold(dimension=2, size=self.patch_len, step=self.stride)

        B, C, N, P = patches.shape
        patches = patches.reshape(B * C, N, P)
        x_mamba_emb = self.patch_emb(patches)

        for layer in self.mamba_layers:
            x_mamba_emb = layer(x_mamba_emb)
        x_mamba_emb = self.norm(x_mamba_emb)

        x_mamba = x_mamba_emb.reshape(B, C, N, self.d_model)
        x_mamba = self.d2p(x_mamba).reshape(B, C, N * self.patch_len) # [32, 7, 96]
        x_mamba = (x_mamba - x_mamba.mean(dim=-1, keepdim=True)) / \
               (x_mamba.std(dim=-1, keepdim=True, unbiased=False) + 1e-5)
        
        # === VarDrop 적용 ===
        # 학습 중에만 VarDrop을 적용
        if self.use_vardrop and training:
            # 1. k-DFH를 위해 [B, C, L] -> [B, L, C]로 변경
            # .detach()로 샘플링 과정이 역전파에 영향을 주지 않도록 함
            x_mamba_for_fft = x_mamba.permute(0, 2, 1).detach()
            
            # 2. 대표 변수 인덱스 샘플링
            sample_indices = efficient_sampler(
                x_mamba_for_fft, 
                k=self.k, 
                group_size=self.group_size, 
                freq_list=self.freq_list
            )
            # print(f"===== [VarDrop DEBUG] Selected Indices: {sample_indices} =====")

            # 3. (대안) Sparse Attention Mask 생성
            # 전체 C x C GNN 구조는 유지하되, 선택된 노드들 간의 엣지만 활성화
            x_mamba_corr = cal_corr_matrix(x_mamba) # [B, C, C]
            
            with torch.no_grad():
                mask = torch.zeros(C, C, device=x_mamba.device, dtype=torch.float32)
                # 샘플링된 인덱스들의 교차점(intersection)에만 1을 할당
                idx_tensor = torch.tensor(sample_indices, device=x_mamba.device)
                grid_i, grid_j = torch.meshgrid(idx_tensor, idx_tensor, indexing='ij')
                mask[grid_i, grid_j] = 1.0
            
            # 4. 마스크 적용 (선택된 노드 간의 관계만 남김)
            x_mamba_corr = x_mamba_corr * mask.unsqueeze(0) # [B, C, C]

        else:
            # VarDrop을 사용하지 않거나 추론 시에는 전체 그래프 사용
            x_mamba_corr = cal_corr_matrix(x_mamba) # [B, C, C]

        # 1. 각 그래프의 edge_index와 edge_weight 리스트 생성
        edge_indices = []
        edge_weights = []
        for i in range(B):
            adj_matrix = x_mamba_corr[i]
            threshold = 0.2
            adj_matrix[torch.abs(adj_matrix) < threshold] = 0
            edge_index, edge_weight = dense_to_sparse(adj_matrix)
            
            # 2. 노드 인덱스 오프셋 적용
            # i번째 그래프의 노드 인덱스에 (i * C)만큼 더해줌
            edge_indices.append(edge_index + i * C)
            edge_weights.append(edge_weight)

        # 3. 전체 배치를 위한 단일 edge_index와 edge_weight 생성
        total_edge_index = torch.cat(edge_indices, dim=1)
        total_edge_weight = torch.cat(edge_weights, dim=0)

        # ============ [디버깅 코드: 엣지 수 확인] ============
        # if training:
        #     mode_str = "VarDrop" if self.use_vardrop else "FullGraph"
        #     avg_edges = total_edge_weight.shape[0] / B  # 배치 당 평균 엣지 수
        #     print(f"[{mode_str}] Total Edges: {total_edge_weight.shape[0]}, Avg Edges/Batch: {avg_edges:.2f}")
        # =================================================

        x_mamba = x_mamba.permute(0, 2, 1)
        # 4. STGAT 레이어에 배치 전체를 한 번에 전달
        x_graph_out = self.stgat_layer(
            x=x_mamba,
            edge_index=total_edge_index,
            edge_weight=total_edge_weight
        )
        # 최종 출력 계산 (역정규화)
        x_out = x_graph_out * std + mean

        return x_out
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        if self.task_name in ["short_term_forecast", "long_term_forecast"]:
            # [수정됨] training 플래그를 모델의 상태(self.training)에서 가져옴
            x_out = self.forecast(x_enc, x_mark_enc, training=self.training)
            # 최종 반환 형태에 맞게 슬라이싱
            return x_out
        return None

class VarDropMemoryProfiler:
    """VarDrop 사용 유무에 따른 메모리 사용량 비교 프로파일러"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.results = defaultdict(dict)
    
    def _reset_memory(self):
        """메모리 초기화"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def _get_memory_stats(self):
        """현재 메모리 통계 반환 (GB 단위)"""
        if self.device == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }
        return None
    
    def measure_single_forward(self, x_enc, x_mark_enc, use_vardrop, phase='train'):
        """단일 forward pass 메모리 측정"""
        self._reset_memory()
        
        # VarDrop 설정
        original_vardrop = self.model.use_vardrop
        self.model.use_vardrop = use_vardrop
        
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        # Forward pass
        with torch.set_grad_enabled(phase == 'train'):
            output = self.model(x_enc, x_mark_enc)
        
        mem_stats = self._get_memory_stats()
        
        # 원래 설정으로 복구
        self.model.use_vardrop = original_vardrop
        
        return mem_stats, output
    
    def measure_forward_backward(self, x_enc, x_mark_enc, use_vardrop):
        """Forward + Backward pass 메모리 측정"""
        self._reset_memory()
        
        original_vardrop = self.model.use_vardrop
        self.model.use_vardrop = use_vardrop
        self.model.train()
        
        # Forward
        output = self.model(x_enc, x_mark_enc)
        mem_after_forward = self._get_memory_stats()
        
        # Backward
        loss = output.mean()
        loss.backward()
        mem_after_backward = self._get_memory_stats()
        
        # 원래 설정으로 복구
        self.model.use_vardrop = original_vardrop
        
        return {
            'forward': mem_after_forward,
            'backward': mem_after_backward
        }
    
    def compare_vardrop(self, x_enc, x_mark_enc, num_iterations=5):
        """VarDrop 사용/미사용 비교"""
        print("=" * 70)
        print("VarDrop 메모리/성능 비교 분석")
        print("=" * 70)
        
        # 1. 단일 Forward Pass 비교 (Training Mode)
        print("\n[1] Training Mode - Forward Pass (Memory)")
        print("-" * 70)
        
        without_results = []
        with_results = []
        
        for _ in range(num_iterations):
            mem_without, _ = self.measure_single_forward(x_enc, x_mark_enc, False, 'train')
            without_results.append(mem_without['max_allocated'])
            mem_with, _ = self.measure_single_forward(x_enc, x_mark_enc, True, 'train')
            with_results.append(mem_with['max_allocated'])
        
        avg_without = np.mean(without_results)
        avg_with = np.mean(with_results)
        reduction = ((avg_without - avg_with) / avg_without) * 100
        
        print(f"VarDrop 미사용: {avg_without:.4f} GB (std: {np.std(without_results):.4f})")
        print(f"VarDrop 사용:   {avg_with:.4f} GB (std: {np.std(with_results):.4f})")
        print(f"메모리 절감:    {avg_without - avg_with:.4f} GB ({reduction:.2f}%)")
        
        # 2. Forward + Backward Pass 비교
        print("\n[2] Training Mode - Forward + Backward Pass (Memory)")
        print("-" * 70)
        
        fb_without_results = []
        fb_with_results = []
        
        for _ in range(num_iterations):
            mem_without = self.measure_forward_backward(x_enc, x_mark_enc, False)
            fb_without_results.append(mem_without['backward']['max_allocated'])
            mem_with = self.measure_forward_backward(x_enc, x_mark_enc, True)
            fb_with_results.append(mem_with['backward']['max_allocated'])
        
        avg_fb_without = np.mean(fb_without_results)
        avg_fb_with = np.mean(fb_with_results)
        fb_reduction = ((avg_fb_without - avg_fb_with) / avg_fb_without) * 100
        
        print(f"VarDrop 미사용: {avg_fb_without:.4f} GB (std: {np.std(fb_without_results):.4f})")
        print(f"VarDrop 사용:   {avg_fb_with:.4f} GB (std: {np.std(fb_with_results):.4f})")
        print(f"메모리 절감:    {avg_fb_without - avg_fb_with:.4f} GB ({fb_reduction:.2f}%)")
        
        # 3. Inference Mode 비교
        print("\n[3] Inference Mode - Forward Pass (Memory)")
        print("-" * 70)
        
        inf_without_results = []
        inf_with_results = []
        
        for _ in range(num_iterations):
            mem_without, _ = self.measure_single_forward(x_enc, x_mark_enc, False, 'eval')
            inf_without_results.append(mem_without['max_allocated'])
            mem_with, _ = self.measure_single_forward(x_enc, x_mark_enc, True, 'eval')
            inf_with_results.append(mem_with['max_allocated'])
        
        avg_inf_without = np.mean(inf_without_results)
        avg_inf_with = np.mean(inf_with_results)
        
        print(f"VarDrop 미사용: {avg_inf_without:.4f} GB (std: {np.std(inf_without_results):.4f})")
        print(f"VarDrop 사용:   {avg_inf_with:.4f} GB (std: {np.std(inf_with_results):.4f})")
        print(f"차이:           {avg_inf_without - avg_inf_with:.4f} GB")
        print("(주의: Inference에서는 VarDrop이 적용되지 않음)")
        
        # 4. 배치 크기별 비교
        print("\n[4] 배치 크기별 메모리 절감 효과")
        print("-" * 70)
        self.compare_by_batch_size(x_enc, x_mark_enc)
        
        # ==========================================================
        # === 5. Latency 비교 (torch.profiler) - 수정된 부분 ===
        # ==========================================================
        print("\n[5] 실제 실행 시간(Latency) 비교 (Training Mode, Forward Pass)")
        print("-" * 70)
        
        inputs = (x_enc, x_mark_enc)

        # UnboundLocalError 방지를 위해 변수들을 0으로 초기화
        avg_cuda_time_without = 0
        avg_cuda_time_with = 0
        time_reduction = 0
        time_reduction_pct = 0
        
        try:
            # --- 1. VarDrop 미사용 시 시간 측정 ---
            self.model.use_vardrop = False
            self.model.train()
            
            with torch.autograd.profiler.profile(
                use_device='cuda',
                record_shapes=False
            ) as prof_without:
                for _ in range(num_iterations):
                    with torch.autograd.profiler.record_function("model_forward_without"):
                        _ = self.model(*inputs)
            
            # --- 2. VarDrop 사용 시 시간 측정 ---
            self.model.use_vardrop = True
            self.model.train()
            
            with torch.autograd.profiler.profile(
                use_device='cuda',
                record_shapes=False
            ) as prof_with:
                for _ in range(num_iterations):
                    with torch.autograd.profiler.record_function("model_forward_with"):
                        _ = self.model(*inputs)

            # --- 3. 결과 출력 ---
            event_list_without = prof_without.key_averages()
            event_without = [e for e in event_list_without if e.key == "model_forward_without"]
            if event_without:
                # [FIX 3] AttributeError 해결: cuda_time_total -> cuda_time
                avg_cuda_time_without = (event_without[0].cuda_time / num_iterations) / 1000 # ms
            
            event_list_with = prof_with.key_averages()
            event_with = [e for e in event_list_with if e.key == "model_forward_with"]
            if event_with:
                # [FIX 3] AttributeError 해결: cuda_time_total -> cuda_time
                avg_cuda_time_with = (event_with[0].cuda_time / num_iterations) / 1000 # ms

            if avg_cuda_time_without > 0: # 0으로 나누기 방지
                time_reduction = avg_cuda_time_without - avg_cuda_time_with
                time_reduction_pct = (time_reduction / avg_cuda_time_without) * 100

            print(f"VarDrop 미사용: {avg_cuda_time_without:.4f} ms (GPU 실행 시간, 평균)")
            print(f"VarDrop 사용:   {avg_cuda_time_with:.4f} ms (GPU 실행 시간, 평균)")
            print(f"실행 시간 절감: {time_reduction:.4f} ms ({time_reduction_pct:.2f}%)")
        
        except AttributeError as e:
            print(f"프로파일링 중 오류 발생 (AttributeError): {e}")
            print("PyTorch 프로파일러 API가 호환되지 않습니다. 'cuda_time' 속성을 확인하세요.")
        except Exception as e:
            print(f"프로파일링 중 예외 발생: {e}")
            print("GPU/CUDA 환경이 올바른지 확인하세요.")
        
        print("\n" + "=" * 70)
        
        return {
            'forward_only': {
                'without_vardrop': avg_without,
                'with_vardrop': avg_with,
                'reduction_gb': avg_without - avg_with,
                'reduction_pct': reduction
            },
            'forward_backward': {
                'without_vardrop': avg_fb_without,
                'with_vardrop': avg_fb_with,
                'reduction_gb': avg_fb_without - avg_fb_with,
                'reduction_pct': fb_reduction
            },
            'latency': {
                'without_vardrop_ms': avg_cuda_time_without,
                'with_vardrop_ms': avg_cuda_time_with,
                'reduction_ms': time_reduction,
                'reduction_pct': time_reduction_pct
            }
        }
    
    def compare_by_batch_size(self, x_enc, x_mark_enc):
        """배치 크기별 메모리 비교"""
        original_batch_size = x_enc.shape[0]
        batch_sizes = [8, 16, 32, 64] if original_batch_size >= 64 else [4, 8, 16, 32]
        
        print(f"{'Batch':<8} {'Without VarDrop':<18} {'With VarDrop':<18} {'Reduction':<15}")
        print("-" * 70)
        
        for bs in batch_sizes:
            try:
                # [수정됨] x_mark_enc가 None일 경우 처리
                if bs <= original_batch_size:
                    x_batch = x_enc[:bs]
                    x_mark_batch = x_mark_enc[:bs] if x_mark_enc is not None else None
                else:
                    repeats = (bs + original_batch_size - 1) // original_batch_size
                    x_batch = x_enc.repeat(repeats, 1, 1)[:bs]
                    x_mark_batch = x_mark_enc.repeat(repeats, 1, 1)[:bs] if x_mark_enc is not None else None
                
                # VarDrop 없이
                self._reset_memory()
                mem_without, _ = self.measure_single_forward(x_batch, x_mark_batch, False, 'train')
                
                # VarDrop 사용
                self._reset_memory()
                mem_with, _ = self.measure_single_forward(x_batch, x_mark_batch, True, 'train')
                
                reduction = mem_without['max_allocated'] - mem_with['max_allocated']
                reduction_pct = (reduction / mem_without['max_allocated']) * 100 if mem_without['max_allocated'] > 0 else 0
                
                print(f"{bs:<8} {mem_without['max_allocated']:>8.4f} GB      "
                      f"{mem_with['max_allocated']:>8.4f} GB      "
                      f"{reduction:>6.4f} GB ({reduction_pct:>5.2f}%)")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"{bs:<8} OOM Error")
                    break
                else:
                    raise e


# ============================================================================
# 사용 예제 코드
# ============================================================================

def run_vardrop_memory_test(model, x_enc, x_mark_enc=None, num_iterations=5):
    """
    VarDrop 메모리/성능 테스트를 실행하는 간단한 함수
    
    Args:
        model: 측정할 모델 (Model 클래스 인스턴스)
        x_enc: 입력 데이터 [batch_size, seq_len, enc_in]
        x_mark_enc: 시간 마크 (선택사항)
        num_iterations: 각 측정당 반복 횟수
    
    Returns:
        결과 딕셔너리
    """
    profiler = VarDropMemoryProfiler(model, device='cuda' if torch.cuda.is_available() else 'cpu')
    # [수정됨] thop_profile 인자 제거
    results = profiler.compare_vardrop(x_enc, x_mark_enc, num_iterations=num_iterations)
    return results


if __name__ == "__main__":
    
    print("=" * 70)
    print("VarDrop 메모리/성능 프로파일러")
    print("=" * 70)
    print("\n테스트 시작...")
    print("-" * 70)

    # 1. 'configs' 객체를 생성
    my_configs = argparse.Namespace(
        task_name="long_term_forecast",
        seq_len=96,     # x_enc의 L (96)과 일치
        pred_len=96,    # stgat_layer에 필요
        d_model=128,    # Mamba 레이어 차원
        enc_in=7,       # x_enc의 C (7)와 일치
        embed='timeF',
        freq='h',
        patch_len=16,   # seq_len의 약수
        stride=16,
        use_vardrop=True,
        vardrop_k=3,    # (k < enc_in)
        group_size=1,
        e_layers=2,     # Mamba 레이어 수
        d_state=16,     # Mamba d_state
        d_conv=4,       # Mamba d_conv
        expand=2,       # Mamba expand
        masking_ratio=0.4
    )

    # 2. 모델 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("경고: CUDA를 사용할 수 없습니다. CPU로 실행합니다. (성능/메모리 측정이 정확하지 않음)")

    model = Model(configs=my_configs, mamba_class=Mamba).float().to(device)

    # 3. 입력 데이터 생성
    x_enc = torch.randn(32, 96, 7).to(device)
    x_mark_enc = None # 예시에서는 사용하지 않음

    # 4. 프로파일러 실행
    # (주의: 최초 실행 시 CUDA 커널 로딩 등으로 인해 시간이 더 걸릴 수 있습니다.)
    results = run_vardrop_memory_test(model, x_enc, x_mark_enc, num_iterations=10)

    # 5. 최종 결과 확인
    print("\n--- 최종 요약 ---")
    print(f"Forward만 (Memory):        {results['forward_only']['reduction_pct']:.2f}% 절감")
    print(f"Forward+Backward (Memory): {results['forward_backward']['reduction_pct']:.2f}% 절감")
    print(f"Latency (Train Mode):    {results['latency']['reduction_pct']:.2f}% 절감")