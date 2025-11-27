import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
import torch.optim as optim
import math
from scipy.signal import chirp

# --- 모델 및 유틸리티 클래스 정의 (이전과 동일, 생략하지 않고 모두 포함) ---
try:
    from mamba_ssm import Mamba as Mamba_SSM
except ImportError:
    print("Mamba-ssm이 설치되어 있지 않습니다. 'pip install mamba-ssm causal-conv1d' 명령어로 설치해주세요.")
    Mamba_SSM = None

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class ResidualMambaBlock(nn.Module):
    def __init__(self, mamba_block, d_model):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = mamba_block
    def forward(self, x):
        return self.mamba(self.norm(x)) + x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class SimpleTransformerDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_proj  = nn.Linear(1, cfg.d_model)
        self.pos_enc  = PositionalEncoding(cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads,
                                                   batch_first=True, norm_first=True, 
                                                   activation='gelu', dim_feedforward=cfg.d_model*4)
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=cfg.e_layers)
        self.out_proj = nn.Linear(cfg.d_model, 1)

    def forward(self, x):
        z = self.in_proj(x)
        z = self.pos_enc(z)
        z = self.encoder(z)
        y = self.out_proj(z)
        return y

class SimpleMambaDenoiser(nn.Module):
    def __init__(self, cfg, mamba_class):
        super().__init__()
        self.in_proj = nn.Linear(1, cfg.d_model)
        self.mamba_layers = nn.ModuleList([
            ResidualMambaBlock(
                mamba_class(d_model=cfg.d_model, d_state=cfg.d_state,
                            d_conv=cfg.d_conv, expand=cfg.expand),
                d_model=cfg.d_model
            ) for _ in range(cfg.e_layers)
        ])
        self.out_proj = nn.Linear(cfg.d_model, 1)

    def forward(self, x):
        z = self.in_proj(x)
        for layer in self.mamba_layers:
            z = layer(z)
        y = self.out_proj(z)
        return y

# --- 데이터셋 생성 함수들 ---
def create_chirp_dataset(seq_len):
    time = np.linspace(0, 50, seq_len)
    clean_signal = chirp(time, f0=0.1, f1=1.5, t1=50, method='linear')
    random_noise = 0.5 * np.random.randn(seq_len)
    specific_high_freq_noise = 0.4 * np.sin(20 * time) 
    noisy_signal = clean_signal + random_noise + specific_high_freq_noise
    return noisy_signal, clean_signal

def create_real_world_dataset(file_path, column_name='HUFL', length=1024, smoothing_window=15):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None, None
    original_signal = df[column_name].values[2000:2000+length]
    original_signal = (original_signal - np.mean(original_signal)) / np.std(original_signal)
    clean_signal = pd.Series(original_signal).rolling(window=smoothing_window, center=True, min_periods=1).mean().to_numpy()
    return original_signal, clean_signal

# --- 시각화 함수 (이전과 동일) ---
def plot_final_comparison(noisy, clean_target, tr_pred, mb_pred, title="Denoising Performance"):
    """시간 영역과 주파수 영역 결과를 함께 시각화합니다."""
    signals = {
        'Noisy Input': noisy,
        'Clean Target': clean_target,
        'Transformer Output': tr_pred,
        'Mamba Output': mb_pred
    }
    colors = {'Noisy Input': 'gray', 'Clean Target': 'black',
              'Transformer Output': 'darkorange', 'Mamba Output': 'green'}

    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    fig.suptitle(title + ': Time and Frequency Domain', fontsize=20)

    # --- 시간 영역 플롯 ---
    axes[0].set_title('Time-Domain Waveforms', fontsize=16)
    # (이전 코드와 동일)
    axes[0].plot(signals['Noisy Input'], label='Noisy Input', color=colors['Noisy Input'], alpha=0.6)
    axes[0].plot(signals['Clean Target'], label='Clean Target', color=colors['Clean Target'], linestyle='--', linewidth=2.5)
    axes[0].plot(signals['Transformer Output'], label='Transformer Output', color=colors['Transformer Output'], linewidth=2)
    axes[0].plot(signals['Mamba Output'], label='Mamba Output', color=colors['Mamba Output'], linewidth=2)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 주파수 영역 플롯 (FFT) ---
    axes[1].set_title('Frequency Spectrums (via FFT)', fontsize=16)
    N = len(noisy)
    freq_axis = np.fft.rfftfreq(N) # 샘플링 주파수를 1로 가정

    for name, signal in signals.items():
        fft_magnitude = np.abs(np.fft.rfft(signal)) / N
        axes[1].plot(freq_axis, fft_magnitude, label=name, color=colors[name],
                     alpha=0.9 if name != 'Noisy Input' else 0.5,
                     linewidth=2.5 if name != 'Noisy Input' else 2)

    axes[1].legend(fontsize=12)
    axes[1].set_xlabel("Frequency (Normalized)", fontsize=12)
    axes[1].set_ylabel("Magnitude", fontsize=12)
    axes[1].set_xlim(0, freq_axis.max() / 4) # 저주파~중주파 영역에 집중
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{title.replace(' ', '_').lower()}_with_spectrum.png", dpi=300)
    plt.show()

# --- 메인 실행 스크립트 ---
if __name__ == "__main__":
    if Mamba_SSM is None: exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    configs = SimpleNamespace(d_model=128, e_layers=4, n_heads=8,
                              d_state=16, d_conv=4, expand=2,
                              epochs=1000, learning_rate=0.0005)
    seq_len = 1024

    # --- 실험 선택 (아래 중 하나를 선택하고 나머지는 주석 처리) ---
    EXPERIMENT_TYPE = "ETTh1"  # "CHIRP" 또는 "REAL_WORLD"
    # EXPERIMENT_TYPE = "REAL_WORLD"
    
    if EXPERIMENT_TYPE == "CHIRP":
        train_noisy_signal, train_clean_signal = create_chirp_dataset(seq_len)
        test_noisy_signal, test_clean_signal = create_chirp_dataset(seq_len)
        title = "Denoising Performance on Chirp Signal"
    else:
        train_noisy_signal, train_clean_signal = create_real_world_dataset('/data/whan_i/final_paper/dataset/ETT-small/ETTh1.csv', length=seq_len)
        test_noisy_signal, test_clean_signal = create_real_world_dataset('/data/whan_i/final_paper/dataset/ETT-small/ETTh1.csv', length=seq_len, column_name='HUFL') # 다른 컬럼으로 테스트
        if train_noisy_signal is None: exit()
        title = "Denoising Performance on Real-World Signal"

    # --- 훈련 및 평가 로직 (이전과 동일) ---
    train_noisy = torch.tensor(train_noisy_signal, dtype=torch.float32)[None, :, None].to(device)
    train_clean = torch.tensor(train_clean_signal, dtype=torch.float32)[None, :, None].to(device)
    test_noisy = torch.tensor(test_noisy_signal, dtype=torch.float32)[None, :, None].to(device)

    def train(model):
        model.to(device).train()
        opt = optim.AdamW(model.parameters(), lr=configs.learning_rate)
        loss_fn = nn.MSELoss()
        print(f"\n--- Training {model.__class__.__name__} for {EXPERIMENT_TYPE} ---")
        for ep in range(configs.epochs):
            opt.zero_grad()
            pred = model(train_noisy)
            loss = loss_fn(pred, train_clean)
            loss.backward()
            opt.step()
            if (ep+1) % 400 == 0:
                print(f'Epoch {ep+1}/{configs.epochs}, Loss: {loss.item():.6f}')
        return model

    tr_model = SimpleTransformerDenoiser(configs)
    mb_model = SimpleMambaDenoiser(configs, Mamba_SSM)
    
    trained_tr_model = train(tr_model)
    trained_mb_model = train(mb_model)

    trained_tr_model.eval(); trained_mb_model.eval()
    with torch.no_grad():
        tr_out = trained_tr_model(test_noisy).cpu().squeeze().numpy()
        mb_out = trained_mb_model(test_noisy).cpu().squeeze().numpy()

    mse_tr = np.mean((tr_out - test_clean_signal)**2)
    mse_mb = np.mean((mb_out - test_clean_signal)**2)

    print("\n--- Generalization Performance (Test MSE) ---")
    print(f"Transformer MSE: {mse_tr:.6f}")
    print(f"Mamba MSE:       {mse_mb:.6f}")

    # 최종 결과 시각화 함수를 주파수 분석이 포함된 버전으로 호출
    plot_final_comparison(test_noisy_signal, test_clean_signal, tr_out, mb_out, title=title)