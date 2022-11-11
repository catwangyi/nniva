# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:20:37 2020

Created on 2018/12
Author: Kaituo XU

Edited by: yoonsanghyu  2020/04
"""

import os
import time
import torch
import numpy as np
import logging
from pit_cal_AEC import *
from utility.fea_tool import stft, istft


class Solver(object):
    
    def __init__(self, data, model, optimizer, scheduler, args):
        self.tr_loader  = data['tr_loader']
        self.cv_loader  = data['cv_loader']
        self.model      = model
        self.optimizer  = optimizer
        self.visdom_lr  = None
        self.scheduler  = scheduler
        self.N_fft      = args.N_fft
        self.hop_length = args.hop_length
        self.seq_len    = int(args.segment*16000)
        # Training config
        self.use_cuda   = args.use_cuda
        self.epochs     = args.epochs
        self.half_lr    = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm   = args.max_norm
        # save and load model
        self.save_folder    = args.save_folder
        self.model_name     = args.model_name
        self.batch_size     = args.batch_size
        self.checkpoint     = args.checkpoint
        self.continue_from  = args.continue_from
        # logging
        self.print_freq     = args.print_freq
        self.logger         = logging.getLogger('CLDNN_based_aec')
        # visualizing loss using visdom
        self.visdom         = args.visdom
        self.visdom_epoch   = args.visdom_epoch
        self.visdom_id      = args.visdom_id
        # loss
        self.tr_loss    = torch.Tensor(self.epochs)
        self.cv_loss    = torch.Tensor(self.epochs)
        self.loss_func  = args.loss_func
        self._reset()


    def _reset(self):
        # Reset
        if self.continue_from:
            self.logger.info('Loading checkpoint model %s' % self.continue_from)
            cont = torch.load(self.continue_from)
            self.start_epoch = cont['epoch']
            self.tr_loss=cont['tr_loss']
            self.cv_loss=cont['cv_loss']
            self.model.load_state_dict(cont['model_state_dict'])
            self.optimizer.load_state_dict(cont['optimizer_state'])
            torch.set_rng_state(cont['trandom_state'])
            np.random.set_state(cont['nrandom_state'])
            self.best_val_loss = float("inf")
            self.best_val_loss = self.cv_loss[self.start_epoch-1]
            print('the best_val_loss is: {}'.format(self.best_val_loss))
        else:
            self.start_epoch = 0
            self.best_val_loss = float("inf")
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.halving = False
        self.val_no_impv = 0


    def train(self):
        # Train model multi-epoches
        best_model_path = ""
        result_save_dir = os.path.join(self.save_folder, self.model_name+'_batch'+str(self.batch_size)+'_result')
        os.makedirs(result_save_dir, exist_ok=True)
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("start training...")
            optim_state = self.optimizer.state_dict()
            self.logger.info('epoch start Learning rate: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
            self.logger.info("Training...")
            '''
            #用于自己断点重续动态调整学习率
            if epoch==83:
                self.halving=True
                if self.halving:
                    optim_state = self.optimizer.state_dict()
                    optim_state['param_groups'][0]['lr'] = \
                        optim_state['param_groups'][0]['lr'] / 2.0
                    self.optimizer.load_state_dict(optim_state)
                    self.logger.info('Learning rate adjusted to: {lr:.6f}'.format(
                        lr=optim_state['param_groups'][0]['lr']))
                    self.halving = False
            '''
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss, tr_avg_mse_loss, tr_avg_SI_SNR_loss = self._run_one_epoch(epoch)
            self.logger.info('-' * 85)
            self.logger.info('Train Summary | End of Epoch {0:5d} | Time {1:.2f}s | '
                  'Train Loss {2:.3f} | Train MSELoss {3:.3f} | Train SISNR_Loss {4:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss, tr_avg_mse_loss, tr_avg_SI_SNR_loss))
            self.logger.info('-' * 85)


            # Cross validation
            self.logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            print("start validation...")
            with torch.no_grad():
                val_loss, val_MSE_loss, val_SISNR_loss = self._run_one_epoch(epoch, cross_valid=True)
                
            self.logger.info('-' * 85)
            self.logger.info('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f} | Valid MSELoss {3:.3f} | Valid SISNR_Loss {4:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss, val_MSE_loss, val_SISNR_loss))
            self.logger.info('-' * 85)
            
            #note schedule here
            self.scheduler.step()
            optim_state = self.optimizer.state_dict()
            self.optimizer.load_state_dict(optim_state)
            self.logger.info('Learning rate adjusted to: {lr:.6f}'.format(
                        lr=optim_state['param_groups'][0]['lr']))
            
            # Adjust learning rate (halving)
            #note 若验证集三次没更新，学习率就降一半，若10次没更新就停止
            if self.half_lr:
                if val_loss >= self.best_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 4: #需要多给年轻人一些机会
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        self.logger.info("No imporvement for 10 epochs, early stopping.")
                        break # 这里break的是外面的大循环
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                self.logger.info('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False

            # Save model each epoch
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            '''
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save({
                    'epoch': epoch+1,
                    'tr_loss':self.tr_loss,
                    'cv_loss':self.cv_loss,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, file_path)
                self.logger.info('Saving checkpoint model to %s' % file_path)
            '''
            # Save the best model
            if val_loss < self.best_val_loss:
                
                self.best_val_loss = val_loss
                # model directly save                 
                # model_filename = self.model_name + '_batch' + str(self.batch_size) + '_epoch' + str(epoch+1) + '.pkl'
                # best_model_path = os.path.join(self.save_folder, model_filename)
                # torch.save(self.model, best_model_path)

                # model save format by zhenyu
                best_file_path = os.path.join(self.save_folder, self.model_name + '.pth.tar')                
                # torch.save({
                #     'epoch': epoch+1,
                #     'tr_loss':self.tr_loss,
                #     'cv_loss':self.cv_loss,
                #     'model_state_dict': self.model.state_dict(),
                #     'optimizer_state': self.optimizer.state_dict(),
                #     'trandom_state': torch.get_rng_state(),
                #     'nrandom_state': np.random.get_state()}, best_file_path)
                # self.logger.info("Find better validated model, saving to %s" % best_file_path)

                # best model file save to the same dir with log or other record like train loss pic
                torch.save({
                    'epoch': epoch+1,
                    'tr_loss':self.tr_loss,
                    'cv_loss':self.cv_loss,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, 
                    os.path.join(result_save_dir, 'temp_best_model.pth.tar'))
                

        # rename the directly saved model 
        # os.rename(best_model_path, os.path.join(self.save_folder, "{}_batch{}_best.pkl".format(self.model_name, self.batch_size)))
        print('train finished')

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        total_si_snr_loss = 0
        total_mse_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        for i, (data) in enumerate(data_loader):
            
            # add chunk for 
            padded_mixture, mixture_lengths, padded_source = data
            # print(padded_mixture.shape)       # torch.Size([32, 3, 128000])
            # print(padded_source.shape)        # torch.Size([32, 1, 128000])

            if self.use_cuda:
                padded_mixture = padded_mixture.cuda() # [B, 2, time]: mic, ref
                mixture_lengths = mixture_lengths.cuda()    # not use
                padded_source = padded_source.cuda() # [B, 5, time]: near, clean1_rir, clean2_rir, clean1, clean2

            estimate_stft, estimate_source, mix_stft = self.model(padded_mixture)
            # print(estimate_stft.shape)      # torch.Size([32, 161, 801])
            # print(estimate_source.shape)    # torch.Size([32, 128000])
            # print(mix_stft.shape)           # torch.Size([32, 3, 161, 801, 2])
            
            # mic_stft = mix_stft[:, 0, :, :] # [B, fre, time]
            # ref_stft = mix_stft[:, 1, :, :] # [B, fre, time]

            target_source   = padded_source
            # print(target_source.shape)      # torch.Size([32, 1, 128000])     resever the 1dim for muti target in other task
            estimate_stft   = torch.unsqueeze(estimate_stft, 1)         # torch.Size([32, 1, 161, 801])
            estimate_source = torch.unsqueeze(estimate_source, 1)       # torch.Size([32, 1, 128000])            


            # cal loss, data_shape should be like 
            # 1. [batch_size, nchannel, length] for time_loss
            # 2. [batch_size, nchannel, T, F] for spec_loss
            # 3. [batch_size, nchannel, T, F, 2] for complex_loss
            loss_si_snr = cal_SISNR(estimate_source, target_source)
            if self.loss_func == 'specl1loss':
                target_stft = stft(padded_source, N_fft = self.N_fft, hop_length = self.hop_length)
                estimate_spec = torch.abs(estimate_stft)
                target_spec = torch.abs(target_stft)
                loss_l1loss = cal_l1loss(estimate_spec, target_spec)
                loss        = loss_si_snr + 0.002 * loss_l1loss
            elif self.loss_func == 'timel1loss':
                loss_l1loss = cal_l1loss(estimate_source, target_source)
                loss        = loss_si_snr + 0.002 * loss_l1loss
            elif self.loss_func == 'stftmseloss':
                target_stft = stft(padded_source, N_fft = self.N_fft, hop_length = self.hop_length)
                loss_mse = cal_MSE(estimate_stft, target_stft)
                loss = loss_si_snr + 500.0 * loss_mse
            else:
                print('band ecpp unkown loss')
            
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()
            total_mse_loss += loss_l1loss.item()
            total_si_snr_loss += loss_si_snr.item()
            #这里的loss.item()只是这个Iter的minibatch的平均loss值 
            if i % self.print_freq == 0:
                #optim_state = self.optimizer.state_dict()
                #print('Learning rate adjusted to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
                self.logger.info('Epoch {0:3d} | Iter {1:5d} | Average Loss {2:3.3f} | '
                      'Average MSE_Loss {3:3.6f} | Average SI-SNR_Loss {4:3.6f} | Current Loss {5:3.6f} | {6:5.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1), total_mse_loss / (i + 1), total_si_snr_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)))
                print('Epoch {0:3d} | Iter {1:5d} | Average Loss {2:3.3f} | '
                      'Average MSE_Loss {3:3.6f} | Average SI-SNR_Loss {4:3.6f} | Current Loss {5:3.6f} | {6:5.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1), total_mse_loss / (i + 1), total_si_snr_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)))
        return total_loss / (i + 1), total_mse_loss / (i + 1), total_si_snr_loss / (i + 1)