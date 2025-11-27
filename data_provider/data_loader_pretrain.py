import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


# class Dataset_ETT_hour(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='M', data_path='ETTh1.csv',
#                  target='OT', scale=True):
        
#         # size: [seq_len, label_len(사용안함), pred_len]
#         self.seq_len = size[0]
#         self.pred_len = size[2]
        
#         # 사전 학습에서는 train 데이터만 사용합니다.
#         assert flag == 'train', 'Pre-training only uses the training dataset.'
        
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

#         # === ✨ 핵심 수정 부분: ETT 데이터셋의 표준 Train 분할 방식 적용 ===
#         # 훈련(train) 데이터의 경계를 정의합니다. (12개월치 데이터)
#         border1 = 0
#         border2 = 12 * 30 * 24
#         # ==========================================================

#         # 특성(feature) 선택
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         # 데이터 스케일링
#         if self.scale:
#             # scaler를 학습시킬 때도 동일한 train 경계를 사용합니다.
#             train_data = df_data[border1:border2]
#             self.scaler.fit(train_data.values)
#             # scaler는 전체 데이터에 적용합니다. (올바른 방식)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
        
#         # 최종적으로 사용할 학습 데이터는 정의된 train 경계만큼만 잘라냅니다.
#         self.data_x = data[border1:border2]

#     def __getitem__(self, index):
#         # 1. 시작 지점부터 'seq_len + pred_len' 만큼의 연속된 데이터를 자릅니다.
#         s_begin = index
#         s_end = s_begin + self.seq_len + self.pred_len
        
#         # 2. 이 하나의 연속된 시퀀스가 "시험지와 답안지"의 원본 소스가 됩니다.
#         seq_x = self.data_x[s_begin:s_end]
        
#         # 3. 모델의 pretrain_forward 함수는 x_enc 하나만 필요하므로,
#         #    seq_x와 시간 정보(mark) 없이 데이터만 반환합니다.
#         return (seq_x,)

#     def __len__(self):
#         # 생성 가능한 총 샘플 개수를 계산합니다.
#         return len(self.data_x) - (self.seq_len + self.pred_len) + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

# class Dataset_ETT_minute(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='M', data_path='ETTm1.csv',
#                  target='OT', scale=True): # ✨ timeenc, freq 등 불필요한 인자 제거

#         self.seq_len = size[0]
#         self.pred_len = size[2]
#         assert flag == 'train', 'Pre-training only uses the training dataset.'
        
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

#         # ✨ 시작: ETT minute 훈련 데이터 경계 (12개월) 적용
#         border1 = 0
#         border2 = 12 * 30 * 24 * 4
#         # ✨ 끝

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1:border2]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         self.data_x = data[border1:border2]

#     # ✨ 시작: __getitem__ 메서드 수정
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len + self.pred_len
#         seq_x = self.data_x[s_begin:s_end]
#         return (seq_x,)
#     # ✨ 끝

#     # ✨ 시작: __len__ 메서드 수정
#     def __len__(self):
#         return len(self.data_x) - (self.seq_len + self.pred_len) + 1
#     # ✨ 끝

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='M', data_path='ETTh1.csv',
#                  target='OT', scale=True): # ✨ timeenc, freq 등 불필요한 인자 제거

#         self.seq_len = size[0]
#         self.pred_len = size[2]
#         assert flag == 'train', 'Pre-training only uses the training dataset.'
        
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
        
#         # ✨ 시작: Custom 데이터셋 훈련 데이터 경계 (70%) 적용
#         num_train = int(len(df_raw) * 0.7)
#         border1 = 0
#         border2 = num_train
#         # ✨ 끝

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1:border2]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         self.data_x = data[border1:border2]

#     # ✨ 시작: __getitem__ 메서드 수정
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len + self.pred_len
#         seq_x = self.data_x[s_begin:s_end]
#         return (seq_x,)
#     # ✨ 끝

#     # ✨ 시작: __len__ 메서드 수정
#     def __len__(self):
#         return len(self.data_x) - (self.seq_len + self.pred_len) + 1
#     # ✨ 끝

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


'''masked modeling data_provider for pretrain'''

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, percent=100):
        
        # size: [seq_len, label_len(사용안함), pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # 사전 학습에서는 train 데이터만 사용합니다.
        assert flag == 'train', 'Pre-training only uses the training dataset.'
        
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.percent = percent
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # === ✨ 핵심 수정 부분: ETT 데이터셋의 표준 Train 분할 방식 적용 ===
        # 훈련(train) 데이터의 경계를 정의합니다. (12개월치 데이터)
        border1 = 0
        original_train_len = 12 * 30 * 24
        border2 = border1 + int(original_train_len * (self.percent / 100))
        # ==========================================================

        # 특성(feature) 선택
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 데이터 스케일링
        if self.scale:
            # scaler를 학습시킬 때도 동일한 train 경계를 사용합니다.
            train_data = df_data[border1:border2]
            self.scaler.fit(train_data.values)
            # scaler는 전체 데이터에 적용합니다. (올바른 방식)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 최종적으로 사용할 학습 데이터는 정의된 train 경계만큼만 잘라냅니다.
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        # 1. 시작 지점부터 'seq_len + pred_len' 만큼의 연속된 데이터를 자릅니다.
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # 2. 이 하나의 연속된 시퀀스가 "시험지와 답안지"의 원본 소스가 됩니다.
        seq_x = self.data_x[s_begin:s_end]
        
        # 3. 모델의 pretrain_forward 함수는 x_enc 하나만 필요하므로,
        #    seq_x와 시간 정보(mark) 없이 데이터만 반환합니다.
        return (seq_x,)

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTm1.csv',
                 target='OT', scale=True, percent=100): # ✨ timeenc, freq 등 불필요한 인자 제거

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag == 'train', 'Pre-training only uses the training dataset.'
        
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.percent = percent
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ✨ 시작: ETT minute 훈련 데이터 경계 (12개월) 적용
        border1 = 0
        original_train_len = 12 * 30 * 24 * 4
        border2 = border1 + int(original_train_len * (self.percent / 100))
        # ✨ 끝

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1:border2]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]

    # ✨ 시작: __getitem__ 메서드 수정
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        return (seq_x,)
    # ✨ 끝

    # ✨ 시작: __len__ 메서드 수정
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1
    # ✨ 끝

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, percent=100): # ✨ timeenc, freq 등 불필요한 인자 제거

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag == 'train', 'Pre-training only uses the training dataset.'
        
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.percent = percent
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # ✨ 시작: Custom 데이터셋 훈련 데이터 경계 (70%) 적용
        num_train = int(len(df_raw) * 0.7)
        border1 = 0
        border2 = int(num_train * (self.percent / 100))
        # ✨ 끝

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1:border2]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]

    # ✨ 시작: __getitem__ 메서드 수정
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        return (seq_x,)
    # ✨ 끝

    # ✨ 시작: __len__ 메서드 수정
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1
    # ✨ 끝

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
