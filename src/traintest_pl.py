# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

# not rely on supervised feature

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import random

from tqdm import tqdm

import torch
from torch.cuda.amp import GradScaler, autocast

def train_pl(audio_model, train_loader, args, noise_to_audio=False, noise_to_vision=False):
    
    if not noise_to_audio and not noise_to_vision:
        save_path = f"{args.exp_dir}/complete.pth"
    if noise_to_audio and not noise_to_vision:
        save_path = f"{args.exp_dir}/audio_only.pth"
    if not noise_to_audio and noise_to_vision:
        save_path = f"{args.exp_dir}/vision_only.pth"
    if noise_to_audio and noise_to_vision:
        save_path = f"{args.exp_dir}/noise_to_both.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on ' + str(device))
    torch.set_grad_enabled(True)

    # Metric 초기화
    batch_time, per_sample_time, data_time, per_sample_data_time, loss_meter, per_sample_dnn_time = (
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    )
    
    progress = []
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    best_loss = np.inf  # ✅ Best Loss 초기화
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
        with open(f"{exp_dir}/progress.pkl", "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)

    # 모든 파라미터 Freeze
    print('All model parameters are frozen.')
    for param in audio_model.parameters():
        param.requires_grad = False
        
    audio_model.module.additional_token.requires_grad = True

    # 학습 가능한 파라미터 출력
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {}'.format(sum(p.numel() for p in trainables)))
    for name, param in audio_model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}, shape: {param.shape}")
            
    optimizer = torch.optim.Adam([audio_model.module.additional_token], lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))    

    # Learning Rate Scheduler 설정
    if args.lr_adapt:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Adaptive learning rate scheduler enabled.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
                                                         gamma=args.lrscheduler_decay)
        print(f'LR scheduler starts at {args.lrscheduler_start} epoch, decay rate {args.lrscheduler_decay} every {args.lrscheduler_step} epochs.')

    # 손실 함수 설정
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    epoch += 1
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    print(f"Current #steps={global_step}, #epochs={epoch}")
    print("Start training...")
    result = np.zeros([args.n_epochs, 4])
    audio_model.train()

    total_start_time = time.time()

    while epoch < args.n_epochs + 1:
        epoch_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}")
        end_time = time.time()

        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print(f"Current #epochs={epoch}, #steps={global_step}")

        for i, (a_input, v_input, labels) in enumerate(epoch_loader):
            B = int(args.proportion * a_input.size(0))
            a_input, v_input, labels = a_input[:B], v_input[:B], labels[:B]

            a_input, v_input = a_input.to(device), v_input.to(device)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            with autocast():
                audio_output = audio_model(a_input, v_input, args.ftmode)
                loss = loss_fn(audio_output, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time) / a_input.shape[0])

            # tqdm 업데이트
            epoch_loader.set_postfix({
                'Loss': f'{loss_meter.val:.4f}',
                'Avg Loss': f'{loss_meter.avg:.4f}',
                'Per Sample Time': f'{per_sample_time.avg:.5f}s'
            })

            if np.isnan(loss_meter.avg):
                print("Training diverged due to NaN loss. Stopping training...")
                exit()

            # Best 모델 저장
            if loss_meter.avg < best_loss:
                best_loss = loss_meter.avg
                additional_token = audio_model.module.additional_token if isinstance(audio_model, nn.DataParallel) else audio_model.additional_token

                torch.save(additional_token.detach().cpu(), save_path)
                print(f"Best model updated. Additional token saved at {save_path}")

            end_time = time.time()
            global_step += 1
        epoch += 1  # Epoch 증가

    # ✅ 총 학습 시간 출력
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n🎉 Total Training Time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s 🎉")


def apply_noise_to_batch(batch_fbank, batch_image, noise_params):
    both_noise = noise_params.get("noise_to_audio", False) and noise_params.get("noise_to_vision", False)
    
    # 노이즈 강도 조정 (둘 다 True면 줄임)
    audio_gaussian_std = 3.0 if not both_noise else 1.0
    vision_gaussian_std = 1.5 if not both_noise else 0.5
    vision_blur_kernel = 51 if not both_noise else 15
    vision_pixelate_factor = 0.05 if not both_noise else 0.3

    # 오디오 노이즈
    if noise_params.get("noise_to_audio", False):
        for i in range(batch_fbank.size(0)):
            noise_type = random.choices(['none', 'random', 'gaussian', 'shift'], [0.3, 0.2, 0.4, 0.1])[0]
            if noise_type == 'none':
                batch_fbank[i, :, :] = 0  # 전체 0으로 초기화
            elif noise_type == 'random':
                batch_fbank[i, :, :] += torch.rand_like(batch_fbank[i, :, :], device=batch_fbank.device) * np.random.uniform(0.5, 1.5)
            elif noise_type == 'gaussian':
                batch_fbank[i, :, :] += torch.normal(mean=0.0, std=audio_gaussian_std, size=batch_fbank[i, :, :].size(), device=batch_fbank.device)
            elif noise_type == 'shift':
                shift_value = np.random.randint(-batch_fbank.size(2), batch_fbank.size(2))  # 더 큰 시프트
                batch_fbank[i, :, :] = torch.roll(batch_fbank[i, :, :], shifts=shift_value, dims=1)

    # 비주얼 노이즈
    if noise_params.get("noise_to_vision", False):
        for i in range(batch_image.size(0)):
            noise_type = random.choices(['none', 'gaussian', 'blur', 'pixelate'], [0.2, 0.3, 0.3, 0.2])[0]
            if noise_type == 'none':
                batch_image[i, :, :, :] = 0  # 전체 0으로 초기화
            elif noise_type == 'gaussian':
                batch_image[i] += torch.normal(mean=0.0, std=vision_gaussian_std, size=batch_image[i].size(), device=batch_image.device)
                batch_image[i] = torch.clamp(batch_image[i], -3, 3)  # 클램핑으로 데이터 범위 제한
            elif noise_type == 'blur':
                blur_kernel = torch.ones((3, 1, vision_blur_kernel, vision_blur_kernel), device=batch_image.device) / (vision_blur_kernel ** 2)
                batch_image[i:i+1] = torch.nn.functional.conv2d(batch_image[i:i+1], blur_kernel, padding=vision_blur_kernel // 2, groups=3)
            elif noise_type == 'pixelate':
                height, width = batch_image[i].size(1), batch_image[i].size(2)
                small_image = torch.nn.functional.interpolate(batch_image[i:i+1], scale_factor=vision_pixelate_factor, mode='bilinear')
                batch_image[i:i+1] = torch.nn.functional.interpolate(small_image, size=(height, width), mode='nearest')

    return batch_fbank, batch_image


def validate_pl(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc = [], [], [], [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, _) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            # additional_token을 입력 앞에 추가 (4개 모두 concat)
            complete_token = torch.load(f"{args.exp_dir}/complete.pth").to(device)
            audio_only_token = torch.load(f"{args.exp_dir}/audio_only.pth").to(device)
            vision_only_token = torch.load(f"{args.exp_dir}/vision_only.pth").to(device)
            noise_to_both_token = torch.load(f"{args.exp_dir}/noise_to_both.pth").to(device)
            tokens = [complete_token, audio_only_token, vision_only_token, noise_to_both_token]
            a_input = torch.cat([token.expand(a_input.size(0), -1) for token in tokens], dim=1)
            v_input = torch.cat([token.expand(v_input.size(0), -1) for token in tokens], dim=1)
            
            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae.append(loss_mae.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_mae = np.mean(A_loss_mae)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)

    return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc

def save_data(a_input, v_input, filename):
    with open(filename, 'wb') as f:
        pickle.dump((a_input, v_input), f)
    print(f"Data saved to {filename}")