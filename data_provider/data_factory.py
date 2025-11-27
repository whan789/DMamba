import torch
from torch.utils.data import DataLoader
from data_provider.uea import collate_fn

# ✨ 시작: 데이터 로더 import 수정
# 1. 지도 학습 및 파인튜닝을 위한 기본 데이터 로더
from data_provider.data_loader import (Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader,
                                     MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader)
# 2. 사전 학습을 위한 전용 데이터 로더 (별칭으로 불러와 충돌 방지)
from data_provider.data_loader_pretrain import (Dataset_ETT_hour as Dataset_ETT_hour_Pretrain,
                                                Dataset_ETT_minute as Dataset_ETT_minute_Pretrain,
                                                Dataset_Custom as Dataset_Custom_Pretrain)
# ✨ 끝

def data_provider(args, flag):
    percent = getattr(args, 'data_percent', 100)
    
    # ✨ 시작: stage 값에 따라 사용할 데이터 로더 딕셔너리를 선택
    if args.stage == 'pretrain':
        flag = 'train'  # 사전 학습은 항상 train 모드
        data_dict = {
            'ETTh1': Dataset_ETT_hour_Pretrain,
            'ETTh2': Dataset_ETT_hour_Pretrain,
            'ETTm1': Dataset_ETT_minute_Pretrain,
            'ETTm2': Dataset_ETT_minute_Pretrain,
            'custom': Dataset_Custom_Pretrain,
        }
        # 만약 전용 pretrain 로더가 없는 데이터셋의 경우, 범용 pretrain 로더를 사용
        Data = data_dict[args.data]
    else:
        # 기존 지도 학습 및 파인튜닝 로직
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'custom': Dataset_Custom,
            'm4': Dataset_M4,
            'PSM': PSMSegLoader,
            'MSL': MSLSegLoader,
            'SMAP': SMAPSegLoader,
            'SMD': SMDSegLoader,
            'SWAT': SWATSegLoader,
            'UEA': UEAloader,
        }
        Data = data_dict[args.data]
    # ✨ 끝

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True if flag == 'train' else False
        batch_size = args.batch_size
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        # pretrain stage일 때 size 인자가 올바르게 전달되도록 수정
        size = [args.seq_len, args.label_len, args.pred_len]
        
        # ✨ 시작: pretrain 데이터 로더는 seasonal_patterns 인자를 받지 않으므로 분기 처리
        if args.stage == 'pretrain':
             data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=size,
                features=args.features,
                target=args.target,
                scale=True, # pretrain은 scale을 True로 가정
                percent=percent # ✨ percent 인자 추가
            )
        else:
            if args.data == 'm4':
                drop_last = False
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=size,
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                percent=percent, # ✨ percent 인자 추가
                seasonal_patterns=args.seasonal_patterns
            )
        # ✨ 끝
            
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader