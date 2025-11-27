# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Model(nn.Module):
    def __init__(      
        self,
        d_model,        # feature의 임베딩 차원
        d_state=16,     # 각 채널별 hidden state 크기 (논문에서 h_t)
        d_conv=4,       # kernel_size (local receptive field)
        expand=2,       # SSM module서 hidden layer
# ============= dt 관련 파라미터 ================
        dt_rank="auto",     # delta (time-step gating factor를 만들기 위한 low-rank 표현의 차원 수)
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,       # delta의 초기 표준편차를 얼마나 크게 줄지 결정하는 scaling factor ?????
        dt_init_floor=1e-4,
# ============================================
        conv_bias=True,
        bias=False,     # in_proj, out_proj, x_proj에 bias항을 포함시킬지 말지를 결정
        use_fast_path=True,  # Fused kernel options -> True면 kernel fusion을 사용해 훨씬 빠르게 실행됨
        layer_idx=None,     # inference 중 hidden state를 layer별로 구분, 캐싱하기 위한 고유 식별자
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()      # 부모 클래스 nn.Module의 생성자 호출
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # d_inner = 2 * d_model 
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,        # 입력채널(=d_inner)별로 따로따로 convolution 수행 -> depthwise convolution
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # x_proj : delta, B, C를 만들어내는 projection layer
        # conv _ siLU를 통과한 결과(x)를 받아서, delta 생성용 low-rank 벡터 dt, recurrence weight, B, readout weight c를 출력하는 역할
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # low-rank delta 표현 dt -> full-rank delta gating 벡터로 변환하는 linear layer
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        # delta가 학습 초기부터 적절한 분산을 갖도록 dt_proj.weight을 정교하게 초기화하는 것
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        # 모든 weight을 같은 값으로 설정 -> 실험적으로 delta 분산이 작게 시작됨
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        # 균등분포에서 sampling -> delta 값이 적당히 다양하게 퍼지도록 유도
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # 행렬 A의 log 형태 log A를 초기화하는 부분

        # A를 d_inner개 채널로 복제 -> A : d_inner x d_state 차원 (각 채널마다 동일한 A eigenvalue 세트)
        # 초기값만 같고, 학습이 진행되면 채널마다 다른 값을 가짐
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        # 해당 파라미터에 대해 weight decay(L2 정규화)를 적용하지 말라는 속성
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        # y_t를 생성할 때 ssm 모듈의 출력값에 더할 skip connection 벡터 초기화
        # y_t = C^t * h_t + D * x_t
        # 왜 학습 가능한 파라미터로 설정했을까? 입력의 영향도를 채널별로 조절할 수 있게 하기 위해서
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # conv_state : conv1d에서 직전 시점들 저장
        # ssm_state : h_{t-1}, 즉 selective recurrence 메모리 저장
        # 추론 시에 사용됨 (추론 시에는 auto-regressive 방식으로 진행되므로)
        # 학습은 전체 시퀀스를 한 번에 처리하기 때문에 필요 없음 
        conv_state, ssm_state = None, None

        # inference 중이면,,,
        if inference_params is not None:
            # 이전 시점의 상태(conv_state, ssm_state)를 가져옴
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            # self.in_proj.weight : {d_inner * 2, d_model}
            # in_proj = nn.Linear(d_model, 2*d_inner)
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            # d = d_inner * 2
            "d (b l) -> b d l",
            l=seqlen,
        )
        # 최종적인 xz의 shape : (b, d, l)
        # bias를 broadcasting해서 xz에 더하는 작업
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state) = (d, n)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # 학습 중 & fast path가 가능하면 여기서 전부 fused kernel로 처리
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        # 아니면 unfused로 나눠서 처리
        else:
            # x : SSM recurrence에 들어가는 입력
            # z : gating 신호로 쓰일 벡터
            x, z = xz.chunk(2, dim=1)

            
            # Compute short convolution
            # local receptive field 기반의 feature 추출
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # -> x의 시퀀스 길이 x.shape[-1]가 d_conv보다 작을 경우를 대비한 padding
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                # conv_state를 현재 padded x로 갱신
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

            # causal_conv1d_fn : Mamba 맞춤형 causal 1D convolution
            # 이게 없으면 conv1d
            if causal_conv1d_fn is None:
                # ... : 마지막 차원만 슬라이싱하고, 나머지 차원은 전부 그대로 유지해라
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            # d : mamba block 내부의 hidden representation 차원
            # x_proj : (B, L, d_inner) -> (B, L, dt_rank + 2*d_state)
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            
            # mamba-ssm github기준 dt_rank = 16, d_state = 16
            # dt : (B*L, dt_rank)
            # B : (B*L, d_state)
            # C : (B*L, d_state)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            
            # delta t를 계산하는 단계
            # 원래 dt.shape == (B x L, dt_rank)
            # self.dt_proj.weight @ dt.t() = (D, dt_rank) x (dt_rank, B x L)
            # dt.shape == (D, B x L)
            dt = self.dt_proj.weight @ dt.t()
            # 각 시점마다 delta t를 생성하기 위한 것
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            # selective recurrence 수행
            y = selective_scan_fn(
                x, 
                dt,     # softplus를 통해 과거 반영 정도 결정 
                A,      # 고정된 transition matrix
                B,      # 입력에 따라 시점별로 바뀌는 projection matrix
                C,      # 입력에 따라 시점별로 바뀌는 projection matrix
                self.D.float(),     # skip connection 계수
                z=z,        # gating vector (B, D, L)
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            # autoregressive inference 중일 경우
            if ssm_state is not None:
                y, last_state = y
                # 마지막 시점의 state를 다음 step에 쓰기 위해 저장 
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            # (B, L, D) -> (B, L, d_model)
            out = self.out_proj(y)
        return out

    # step : autoregressive decoding (1 token씩 처리) 모드에서 동작하는 핵심 함수
    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        # hidden.state.shape[1] = L
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        # hidden_states.shape == (B, 1, D)          D = d_model
        # self.in_proj = nn.Linear(d_model, 2 * d_inner)
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        # chunk : 텐서를 균등하게 여러 조각으로 나누는 함수
        # x : SSM의 입력값
        # z : gating으로 사용될 값
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        # triton 기반 커널이 설치되어 있으면 causal_conv1d_update 사용
        # 없는 경우 1D convolution 연산을 수동으로 계산
        if causal_conv1d_update is None:
            # conv_state.shape == (B, D, W)
            # shifts : 해당 차원에서 왼쪽으로 1칸 이동
            # conv_state : 과거 입력 x들을 저장하는 sliding window
            # roll로 과거를 왼쪽으로 한 칸 밂
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            # 현재 입력을 넣어 시점별로 최신 W개의 입력 유지
            conv_state[:, :, -1] = x
            # conv_state.shape == (B, D, W)
            # self.conv1d.weight.shape == (D, 1, W) -> Conv1D에서 각 채널마다 독립적으로 W개의 weight 갖고 있음
            # conv 필터와 입력(conv_state) 간 내적 수행
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            # x.shape == (B, D)
            x = self.act(x).to(dtype=dtype)
        # causal_conv1d_update 고속 커널을 사용할 수 있을 때 pytorch fallback 코드 대신 최적화된 CUDA 연산을 호출하는 부분
        # causal_conv1d_update()
        # : Triton or CUDA로 구현된 커스텀 연산함수로, 한 시점씩 처리하는 causal convolution을 pytorch보다 훨씬 빠르게 수행하도록 설계
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        # x.shape == (B, D)
        # self.x_proj = nn.linear(D, dt_rank + 2 * d_state)
        # x_db.shape == (B, dt_rank + 2*d_state)
        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        # dt.shape == (B, dt_rank) -> (B, d_inner)로 확장
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        # 이전 상태(h_t-1)을 기반으로 현재 출력을 만드는 단계

        # Triton 커널이 없는 경우 -> Pytorch로 recurrence 연산을 직접 계산
        if selective_state_update is None:
            # Discretize A and B
            # dt : 각 채널의 시간 간격
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            # A.shape == (D, d_state)
            # dt를 곱해서 시간 스케일에 맞는 decay factor를 만듦
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            # 입력 gate인 dt와 B를 곱해서, 입력이 상태를 얼마나 업데이트할지 결정
            dB = torch.einsum("bd,bn->bdn", dt, B)
            # 상태 업데이트
            # 이전 상태인 ssm_state를 decay시킴 (x dA)
            # 현재 입력 x를 넣어서 (x dB)
            # 새로운 상태로 갱신
            # h_t = h_{t-1}*dA + x_t*dB
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            # y_t = h_t*C
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            # y = y + sef.D.to(dtype) * x
            y = y + self.D.to(dtype) * x
            print("x shape :", x.shape)
            # Gating 
            y = y * self.act(z)  # (B D)
        # 위 연산을 한꺼번에 커널로 처리함
        # 의미는 완전히 동일, 속도가 훨씬 빠름
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )
        # (B, D) -> (B, d_model)
        out = self.out_proj(y)
        # (B, 1, d_model)
        return out.unsqueeze(1), conv_state, ssm_state


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )  
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
