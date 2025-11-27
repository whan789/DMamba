import argparse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# 파이썬 해시 랜덤 고정
os.environ["PYTHONHASHSEED"] = "0"
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

set_seed(42)

if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(description='Mamba_STGAT')

    # basic config
    # <<< MODIFIED START >>> : 단일 시드를 받도록 수정
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # <<< MODIFIED END >>>
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [CMamba]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/data/whan_i/final_paper/dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multichannel predict multichannel, S:unichannel predict unichannel, MS:multichannel predict unichannel')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='patch stride')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--pretrain_lr', type=float, default=0.001, help='pretraining optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--rev', action='store_true', help='whether to apply RevIN')
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # CMamba
    parser.add_argument('--dt_rank', type=int, default=32)
    parser.add_argument('--patch_num', type=int, default=32)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--dt_min', type=float, default=0.001)
    parser.add_argument('--dt_init', type=str, default='constant', help='random or constant')
    parser.add_argument('--dt_max', type=float, default=0.1)
    parser.add_argument('--dt_scale', type=float, default=1.0)
    parser.add_argument('--dt_init_floor', type=float, default=1e-4)
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--conv_bias', type=bool, default=True)
    parser.add_argument('--pscan', action='store_true', help='use parallel scan mode or sequential mode when training', default=True)
    parser.add_argument('--avg', action='store_true', help='avg pooling', default=False)
    parser.add_argument('--max', action='store_true', help='max pooling', default=False)
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--gddmlp', action='store_true', help='global data-dependent mlp', default=False)
    parser.add_argument('--channel_mixup', action='store_true', help='channel mixup', default=False)
    parser.add_argument('--sigma', type=float, default=1.0)

    # ====================== MTGNN Args ====================== #
    parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes/variables')
    parser.add_argument('--gcn_depth',type=int,default=3,help='graph convolution depth')
    parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--subgraph_size',type=int,default=20,help='k')
    parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
    parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')
    parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
    parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
    parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
    parser.add_argument('--end_channels',type=int,default=128,help='end channels')
    parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
    parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')
    # ====================================================== #

    # loss function
    parser.add_argument('--loss_lambda', type=float, default=0.1, help='similarity loss lambda')
    parser.add_argument('--corr_coef', type=float, default=0.0, help='correlation alignment loss coefficient')

    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')

    # pretrain
    parser.add_argument('--stage', type=str, default='pretrain', help='options: [pretrain, finetune, supervised]')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='pretrain epochs')
    parser.add_argument('--masking_ratio', type=float, default=0.5, help='the ratio of patches masked')
    parser.add_argument('--dec_depth', type=int, default=2, help='number of decoder layers in pretraining')

    
    # fine-tuning
    parser.add_argument('--freeze_epochs', type=int, default=30, help='num of epochs to freeze the encoder during fine-tuning')
    parser.add_argument('--finetune_lr', type=float, default=1e-5, help='specific learning rate for the fine-tuning phase a n4re5ui fter freezing')
    parser.add_argument('--finetune_epochs', type=int, default=30, help='num of epochs for fine-tuning after pretraining')
    parser.add_argument('--corr_thres', type=float, default=0.5, help='num of epochs for fine-tuning after pretraining')


    # Node drop
    parser.add_argument('--use_vardrop', type=str_to_bool, default=False, help='whether to use VarDrop')
    parser.add_argument('--vardrop_k', type=int, default=4, help='num of dominant frequencies to consider in VarDrop')
    parser.add_argument('--group_size', type=int, default=5, help='num of variables to sample from each group in VarDrop')
    parser.add_argument('--topk', type=int, default=4, help='num of variables to sample from each group in VarDrop')

    # data deficiency
    parser.add_argument('--data_percent', type=int, default=100, help='num of variables to sample from each group in VarDrop')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        raise NotImplementedError

    # <<< MODIFIED START >>> : 반복문 제거 및 단일 실행 구조로 변경
    if args.is_training:
        print(f"\n>>>>>>> Running experiment for seed: {args.seed} <<<<<<<")
        set_seed(args.seed)
    
        setting = 'model{}_data{}_seq{}_dp{}_seed{}'.format(
            args.model,
            args.data,
            args.seq_len,
            args.data_path,
            args.seed) # args.seeds -> args.seed

        exp = Exp(args)
        print('>>>>>>> start training : {} >>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if args.stage != 'pretrain':
            print('>>>>>>> testing : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            metrics = exp.test(setting)
            mse = metrics['mse']
            mae = metrics['mae']
            print('\n\n================================================================')
            print(f'Final Results:')
            print(f'MSE: {mse:.4f}')
            print(f'MAE: {mae:.4f}')
            print('================================================================')

        torch.cuda.empty_cache()
    else:
        print(f"\n>>>>>>> Running TEST for seed: {args.seed} <<<<<<<")
        set_seed(args.seed)

        setting = 'model{}_data{}_seq{}_dp{}_seed{}'.format(
            args.model,
            args.data,
            args.seq_len,
            args.data_path,
            args.seed) # args.seeds -> args.seed

        exp = Exp(args)
        print('>>>>>>> testing : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        metrics = exp.test(setting, test=1)
        mse = metrics['mse']
        mae = metrics['mae']
        print('\n\n================================================================')
        print(f'Final Results:')
        print(f'MSE: {mse:.4f}')
        print(f'MAE: {mae:.4f}')
        print('================================================================')
        
        torch.cuda.empty_cache()
    # <<< MODIFIED END >>>

