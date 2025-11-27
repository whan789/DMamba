import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.MTGNN_layer import dilated_inception, mixprop, LayerNorm, graph_constructor

class gtnet(nn.Module):
    def __init__(self, configs,
                 predefined_A=None, static_feat=None, dropout=0.3, node_dim=40, in_dim=1, layers=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = configs.gcn_true    # GCN 사용 여부
        self.gcn_depth = configs.gcn_depth  # GCN Layer 수
        self.num_nodes = configs.enc_in    # 노드의 수 (변수의 수)
        self.dropout = dropout
        self.device = configs.gpu
        self.tanhalpha = configs.tanhalpha  # Graph Learning 시, tanh의 스케일링 파라미터
        self.propalpha = configs.propalpha  # Graph Convolution 시, skip connection의 스케일링 파라미터
        self.out_dim = configs.pred_len
        self.subgraph_size = min(configs.subgraph_size, self.num_nodes) # 부분 그래프의 크기
        self.dilation_exponential=configs.dilation_exponential  # Dilation 증가 비율
        self.conv_channels = configs.conv_channels  # Temporal Convolution의 채널 수 (Residual Block의 내의 F(x)+x에서 F(x)의 채널 수)
        self.residual_channels = configs.residual_channels  # Residual Block의 채널 수 (residual block의 hidden state 크기)
        self.skip_channels = configs.skip_channels  # Skip Connection의 채널 수 (각 블록의 출력은 별도의 컨볼루션 연산을 통해 skip_channels 크기로 변환된 후, 최종 출력에 더해짐)
        self.end_channels = configs.end_channels    # Output Layer의 채널 수 (모든 특징이 종합된 후, 최종 예측값으로 변환되기 직전에 거치는 중간 처리 단계의 크기)
        self.seq_len = configs.seq_len

        # [1. Graph Learning]
        self.buildA_true = configs.buildA_true  # 학습 가능한 그래프 구조 사용 여부
        self.predefined_A = predefined_A    # 사전 정의된 인접 행렬
        
        # Graph Learning을 위한 모듈
        # 데이터로부터 동적으로 노드 간의 관계를 학습해 방향성이 있는 sparse 인접 행렬을 생성하는 역할
        self.gc = graph_constructor(self.num_nodes, self.subgraph_size, node_dim, self.device, alpha=self.tanhalpha, static_feat=static_feat)

        # [2. Temporal Convolution]
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # [3. Graph Convolution]
        self.gconv1 = nn.ModuleList() # GCN (1) ... mixhop-prop으로 구성
        self.gconv2 = nn.ModuleList() # GCN (2) ... mixhop-prop으로 구성
        self.norm = nn.ModuleList() # GCN 통과 후, Layer Normalization
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))     # input 데이터를 가장 먼저 latent space로 임베딩
        
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.residual_channels, 1, self.seq_len))

        self.seq_len = configs.seq_len
        kernel_size = 7
        if self.dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(self.dilation_exponential**layers-1)/(self.dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        # [4. 모듈 추가하기]
        # TC 모듈, GC 모듈을 한 쌍으로 묶어 layer 수만큼 반복적으로 쌓아 올리는 역할
        for i in range(1):
            if self.dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(self.dilation_exponential**layers-1)/(self.dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(self.dilation_exponential**j-1)/(self.dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                # Temporal dependency를 캡처하기 위한 Dilated Convolution
                # filter_convs : x에 어떤 특징이 있는지 추출 (tanh 활성화 함수 사용)
                self.filter_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                # gate_convs : x에서 추출한 특징들이 얼마나 중요한지 판단하는 gate 역할 (sigmoid 활성화 함수 사용)
                self.gate_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_len>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.skip_channels,
                                                    kernel_size=(1, self.seq_len-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                # Spatial dependency를 캡처하기 위한 Graph Convolution
                if self.gcn_true:
                    self.gconv1.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, dropout, self.propalpha))
                    self.gconv2.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, dropout, self.propalpha))

                if self.seq_len>self.receptive_field:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes, self.seq_len - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                # 이전 스테이션보다 더 넓은 수용 영역을 갖도록 dilation 비율 증가
                new_dilation *= self.dilation_exponential

        self.layers = layers

        # [5. Final Convolution]
        # 모든 블록에서 추출한 특징들을 종합해 최종 예측값을 생성하는 역할
        
        # skip connectoin 경로를 통해 종합된 정보(skip_channels)를 입력받아, hiddenl state(end_channels)로 변환
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                             out_channels=self.end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        # end_conv_1의 출력을 입력받아, 최종 예측값(out_dim)으로 변환
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                             out_channels=self.out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        
        # [6. SkipConnection 전/후의 NN]
        # skip connection 경로의 시작과 끝을 처리하는 특수 컨볼루션 레이어
        # 모든 정보가 skip connection 경로에서 원활하게 합쳐질 수 있도록 차원을 맞춰주는 역할
        if self.seq_len > self.receptive_field:

            # 모델에 가장 처음 입력된 원본 데이터를 받아 skip connection 경로의 시작점을 만
            # 입력 시퀀스 전체를 한 번에 요약하여, 시간 축 길이가 1인 전체 요약 정보 생성
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=self.skip_channels, kernel_size=(1, self.seq_len), bias=True)
            # 여러 개의 TC/GC 블록을 거친 후의 출력을 받아 skip connection 경로의 끝점을 만듦
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, self.seq_len-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(self.device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_len, 'input sequence length not equal to preset sequence length'

        if self.seq_len<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_len,0,0,0))



        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                    dense_adp = self.gc.fullA(self.idx)
                else:
                    adp = self.gc(idx)
                    dense_adp = self.gc.fullA(idx)
            else:
                adp = self.predefined_A
                dense_adp = self.predefined_A
        # 입력 데이터를 1x1 컨볼루션을 통해 고차원의 feature space(reisidual_channels)로 임베딩
        # x : (batch_size, residual_channels, num_nodes, seq_len) -> 모델의 주 경로인 residual 경로를 따라 흐르는 정보
        x = self.start_conv(input)

        x = x + self.pos_encoder
        # skip : (batch_size, skip_channels, num_nodes, 1) -> 모델의 부 경로인 skip connection 경로를 따라 흐르는 정보
        # 여러 레이어의 정보를 누적할 변수
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x

            # (1) Temporal Convolution : filter_convs와 gate_convs를 통해 x에서 시계열 특징 추출
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)

            # (2) Skip Connection : (1)에서 추출된 특징을 skip_convs로 처리한 뒤, skip 변수에 더해 누적시킴
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            # (3) Graph Convolution : (1)에서 추출된 특징에 대해 gconv1, gconv2를 적용해 공간적 특징 융합
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            # (4) Residual Connection & Layer Normalization : (3)의 출력을 residual 경로의 입력과 더한 후, 정규화
            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)
        
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        return x, None, dense_adp