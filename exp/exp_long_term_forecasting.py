from data_provider.data_factory import data_provider
from torch.optim import lr_scheduler
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, AverageMeter
from utils.metrics import metric
from utils.losses import VarianceWeightedLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from mamba_ssm import Mamba
from torch.utils.data import Dataset, DataLoader
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.data_loader_pretrain import (Dataset_ETT_hour as Dataset_ETT_hour_Pretrain,
                                                Dataset_ETT_minute as Dataset_ETT_minute_Pretrain,
                                                Dataset_Custom as Dataset_Custom_Pretrain)

warnings.filterwarnings('ignore')

def fisher_z(c, eps=1e-6):
    c = torch.clamp(c, -1+eps, 1-eps)
    return 0.5 * (torch.log1p(c) - torch.log1p(-c)) 


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.model in ['GAEMamba', 'MambaMTGNN', 'MambaMTGNN_masking', "Mamba_stgat", 'Mamba_only', 'STGAT_only']:
            model = self.model_dict[self.args.model].Model(self.args, mamba_class=Mamba).float()
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # ✨ 시작: pretrain stage일 때 전용 데이터 로더를 사용하도록 _get_data 메서드 수정
        if self.args.stage == 'pretrain':
            flag = 'train'
            # 사전 학습용 데이터 로더 딕셔너리
            data_dict_pretrain = {
                'ETTh1': Dataset_ETT_hour_Pretrain,
                'ETTh2': Dataset_ETT_hour_Pretrain,
                'ETTm1': Dataset_ETT_minute_Pretrain,
                'ETTm2': Dataset_ETT_minute_Pretrain,
                'custom': Dataset_Custom_Pretrain,
            }
            Data = data_dict_pretrain.get(self.args.data)

        else:

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
                'UEA': UEAloader
            }
            Data = data_dict[self.args.data]

        if flag == 'test' or flag == 'val':
            shuffle_flag = False
            drop_last = True
            batch_size = self.args.batch_size
        else:  # train
            shuffle_flag = True
            drop_last = True
            batch_size = self.args.batch_size

        data_set = Data(
            root_path=self.args.root_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            data_path=self.args.data_path,
            target=self.args.target,
            scale=True,
            percent=self.args.data_percent
        )

        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss.lower() == 'mae':
            criterion = nn.L1Loss()
        elif self.args.loss.lower() == 'mse':
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                total_loss.update(loss.item(), outputs.shape[0])

        self.model.train()
        return total_loss.avg

    def train(self, setting):
        if self.args.stage == 'pretrain':
            return self.pretrain(setting)
        elif self.args.stage == 'finetune':
            return self.finetune(setting)
        elif self.args.stage == 'supervised':
            return self.supervised_train(setting)
        else:
            print(f"Stage '{self.args.stage}' not recognized, running supervised training by default.")
            return self.supervised_train(setting)

    def pretrain(self, setting):
        print(">>>>>> Start Pre-training Stage <<<<<<")
        train_data, train_loader = self._get_data(flag='train')
        path = os.path.join(self.args.checkpoints, setting, 'pretrain')
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.use_multi_gpu:
            model_module = self.model.module
        else:
            model_module = self.model
        
        # 옵티마이저가 업데이트할 파라미터를 지정
        pretrain_params = [
            {'params': model_module.patch_emb.parameters()},
            {'params': model_module.norm.parameters()},
            {'params': model_module.mamba_encoder.parameters()},
            {'params': model_module.mamba_decoder.parameters()},
            {'params': [model_module.patch_pos_emb]}]
        

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = optim.AdamW(pretrain_params, lr=self.args.pretrain_lr, weight_decay=0.01)
        criterion = self._select_criterion()

        best_loss = float('inf')
        best_model_state = None

        for epoch in range(self.args.pretrain_epochs):
            train_loss = AverageMeter()
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x,) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                
                y_pred, y_true = model_module.pretrain_forward(batch_x)

                loss = criterion(y_pred, y_true)
                
                train_loss.update(loss.item(), batch_x.shape[0])
                loss.backward()
                model_optim.step()

            print(f"Pretrain Epoch: {epoch + 1}, Cost time: {time.time() - epoch_time:.4f}s")
            print(f"Pretrain Epoch: {epoch + 1} | Train Loss: {train_loss.avg:.7f}")

            # Pretraining 과정 중 가장 좋았던 성능의 모델 가중치를 저장하는 역할
            if train_loss.avg < best_loss:
                best_loss = train_loss.avg
                best_model_state = {
                    'patch_emb':model_module.patch_emb.state_dict(),
                    'norm': model_module.norm.state_dict(),
                    'mamba_encoder': model_module.mamba_encoder.state_dict()}
                print(f"Best pretrain loss updated to {best_loss:.7f}. Saving model state...")

            # pretrain 시에는 EarlyStopping이 파일을 저장하지 않도록 임시 수정
            def do_nothing_on_save(*args, **kwargs):
                pass
            early_stopping.save_checkpoint = do_nothing_on_save
            
            early_stopping(train_loss.avg, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        if best_model_state is not None:
            print(f"Saving best model state with loss: {best_loss:.7f}")
            torch.save(best_model_state, os.path.join(path, 'pretrained_encoder.pth'))

        print(">>>>>> Pre-training Finished! <<<<<<")
        return self.model

    def supervised_train(self, setting):
        print(">>>>>> Start Supervised Training Stage <<<<<<")
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting, 'supervised')
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = AverageMeter()
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
            
                train_loss.update(loss.item(), batch_x.shape[0])
                loss.backward()
                model_optim.step()
            
            print(f"Epoch: {epoch + 1}, Cost time: {time.time() - epoch_time:.4f}s")
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss.avg:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def finetune(self, setting):
        print(">>>>>> Start Fine-tuning Stage <<<<<<")
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting, 'finetune')
        if not os.path.exists(path):
            os.makedirs(path)

        pretrain_path = os.path.join(self.args.checkpoints, setting, 'pretrain', 'pretrained_encoder.pth')
        if os.path.exists(pretrain_path):
            print(f"Loading pretrained encoder from {pretrain_path}")
            encoder_weights = torch.load(pretrain_path, map_location=self.device)
            
            model_to_load = self.model.module if self.args.use_multi_gpu else self.model
            
            model_to_load.norm.load_state_dict(encoder_weights['norm'])
            model_to_load.patch_emb.load_state_dict(encoder_weights['patch_emb'])
            model_to_load.mamba_encoder.load_state_dict(encoder_weights['mamba_encoder'])
        else:
            raise FileNotFoundError(f"ERROR: Pretrained encoder weights not found at: {pretrain_path}")
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        criterion = self._select_criterion()

        model_to_handle = self.model.module if self.args.use_multi_gpu else self.model
        pretrained_layers = [model_to_handle.patch_emb, model_to_handle.norm, model_to_handle.mamba_encoder]
        pretrained_params = []
        for layer in pretrained_layers:
            pretrained_params.extend(layer.parameters())

        if self.args.freeze_epochs > 0:
            print(f">>>>>> Freezing encoder for the first {self.args.freeze_epochs} epochs. <<<<<<")
            for param in pretrained_params:
                param.requires_grad = False
            
            head_params = [p for p in self.model.parameters() if not any(id(p) == id(param) for param in pretrained_params)]
            model_optim = optim.AdamW(head_params, lr=self.args.learning_rate)
        else:
            print(">>>>>> Applying differential learning rate from the start. <<<<<<")
            head_params = [p for p in self.model.parameters() if not any(id(p) == id(param) for param in pretrained_params)]
        
            encoder_lr = self.args.finetune_lr
            head_lr = self.args.learning_rate
            print(f"Pretrained Encoder LR: {encoder_lr}, Head LR: {head_lr}")
            
            model_optim = optim.AdamW([
                {'params': pretrained_params, 'lr': encoder_lr},
                {'params': head_params, 'lr': head_lr}
            ])
            
        for epoch in range(self.args.finetune_epochs):
            if self.args.freeze_epochs > 0 and epoch == self.args.freeze_epochs:
                print(f"\n>>>>>> Unfreezing encoder at epoch {epoch + 1} and applying differential LR. <<<<<<")
                for param in pretrained_params:
                    param.requires_grad = True

                head_params = [p for p in self.model.parameters() if not any(id(p) == id(param) for param in pretrained_params)]
                
                encoder_lr = self.args.finetune_lr
                head_lr = self.args.learning_rate
                print(f"Pretrained Encoder LR: {encoder_lr}, Head LR: {head_lr}")

                model_optim = optim.AdamW([
                    {'params': pretrained_params, 'lr': encoder_lr},
                    {'params': head_params, 'lr': head_lr}
                ])

            train_loss = AverageMeter()
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
            
                train_loss.update(loss.item(), batch_x.shape[0])
                loss.backward()
                model_optim.step()
            
            print(f"Epoch: {epoch + 1}, Cost time: {time.time() - epoch_time:.4f}s")
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss.avg:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        batch_size = self.args.batch_size
        self.args.batch_size = 1
        test_data, test_loader = self._get_data(flag='test')
        self.args.batch_size = batch_size
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.extend(pred)
                trues.extend(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.svg'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return {'mse': mse, 'mae': mae}