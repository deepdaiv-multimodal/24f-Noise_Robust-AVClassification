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
import os
import matplotlib.pyplot as plt
import soundfile as sf

from utilities import apply_noise_to_batch

def train_pl(audio_model, train_loader, args, noise_to_audio=False, noise_to_vision=False):
    
    if not noise_to_audio and not noise_to_vision:
        save_path = f"{args.exp_dir}/complete.pth"
    if noise_to_audio and not noise_to_vision:
        save_path = f"{args.exp_dir}/vision_only.pth"
    if not noise_to_audio and noise_to_vision:
        save_path = f"{args.exp_dir}/audio_only.pth"
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
        
    audio_model.module.complete_token.requires_grad = True
    audio_model.module.audio_only_token.requires_grad = True
    audio_model.module.vision_only_token.requires_grad = True
    audio_model.module.noise_to_both_token.requires_grad = True
        

    # 학습 가능한 파라미터 출력
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {}'.format(sum(p.numel() for p in trainables)))
    for name, param in audio_model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}, shape: {param.shape}")
            
    optimizer = torch.optim.Adam(trainables, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))    

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
            a_input, v_input = apply_noise_to_batch(a_input, v_input, {"noise_to_audio": noise_to_audio, "noise_to_vision": noise_to_vision})

            if args.save_data:
                os.makedirs(f"{args.exp_dir}/images", exist_ok=True)
                os.makedirs(f"{args.exp_dir}/audio", exist_ok=True)

                for i in range(a_input.size(0)):
                    # 비전 데이터 저장 (이미지)
                    plt.imsave(f"{args.exp_dir}/images/{global_step}_{i}.png", v_input[i].permute(1, 2, 0).cpu().numpy())

                    # 오디오 데이터 저장 (WAV)
                    audio_path = f"{args.exp_dir}/audio/{global_step}_{i}.wav"
                    save_audio(a_input[i], sample_rate=16000, file_path=audio_path)
            a_input, v_input = a_input.to(device), v_input.to(device)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            with autocast():
                
                if not noise_to_audio and not noise_to_vision:
                    case = 1
                    additional_token = audio_model.module.complete_token
                if noise_to_audio and not noise_to_vision:
                    case = 2
                    additional_token = audio_model.module.vision_only_token
                if not noise_to_audio and noise_to_vision:
                    case = 3
                    additional_token = audio_model.module.audio_only_token
                if noise_to_audio and noise_to_vision:
                    case = 4
                    additional_token = audio_model.module.noise_to_both_token
                
                audio_output = audio_model(a_input, v_input, mode='prompt_learning', case=case)
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


def validate_pl(audio_model, val_loader, args, output_pred=False):
    
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_predictions_pl, A_targets_pl, A_loss_pl = [], [], []
    
    noise_params = {
        "noise_to_audio": args.noise_to_audio if hasattr(args, "noise_to_audio") else False,
        "noise_to_vision": args.noise_to_vision if hasattr(args, "noise_to_vision") else False,
    }
    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)
            
            if noise_params["noise_to_audio"] or noise_params["noise_to_vision"]:
                a_input, v_input = apply_noise_to_batch(a_input, v_input, noise_params)
            
            with autocast():
                audio_output_pl = audio_model(a_input, v_input, 'prompt_inference')
                
            # 결과 수집
            predictions_pl = audio_output_pl.to('cpu').detach()
            A_predictions_pl.append(predictions_pl)
            A_targets_pl.append(labels.to('cpu'))
            
            # predictions = audio_output.to('cpu').detach()
            # A_preditions.append(predictions)
            # A_targets.append(labels.to('cpu'))
            
            # 손실 계산
            loss_pl = args.loss_fn(audio_output_pl, labels)
            A_loss_pl.append(loss_pl.to('cpu').detach())
            
            # loss = args.loss_fn(audio_output, labels)
            # A_loss.append(loss.to('cpu').detach())

            # 배치 시간 업데이트
            batch_time.update(time.time() - end)
            end = time.time()

        # 전체 결과를 병합
        audio_output_pl = torch.cat(A_predictions_pl)
        target_pl = torch.cat(A_targets_pl)
        loss_pl = np.mean(A_loss_pl)

        # audio_output = torch.cat(A_preditions)
        # target = torch.cat(A_targets)
        # loss = np.mean(A_loss)
        
        # 통계 계산
        stats_pl = calculate_stats(audio_output_pl, target_pl)
        # stats = calculate_stats(audio_output, target)

    if output_pred == False:
        return stats_pl, loss_pl
    else:
        # multi-frame 평가를 위해 prediction과 target 반환
        return stats_pl, audio_output_pl, target_pl

def save_audio(audio_tensor, sample_rate, file_path):
    """오디오 텐서를 WAV 파일로 저장 (soundfile 사용)"""
    # (채널 수, 샘플 수) 형태로 변환
    audio_np = audio_tensor.squeeze().cpu().numpy()
    sf.write(file_path, audio_np, sample_rate)