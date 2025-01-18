# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

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
import matplotlib.pyplot as plt

from utilities import apply_noise_to_batch

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, loss_meter, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)

    # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
                'mlp_head2.0.weight', 'mlp_head2.0.bias', 'mlp_head2.1.weight', 'mlp_head2.1.bias',
                'mlp_head_a.0.weight', 'mlp_head_a.0.bias', 'mlp_head_a.1.weight', 'mlp_head_a.1.bias',
                'mlp_head_v.0.weight', 'mlp_head_v.0.bias', 'mlp_head_v.1.weight', 'mlp_head_v.1.bias',
                'mlp_head_concat.0.weight', 'mlp_head_concat.0.bias', 'mlp_head_concat.1.weight', 'mlp_head_concat.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]

    # if freeze the pretrained parameters and only train the newly initialized model (linear probing)
    if args.freeze_base == True:
        print('Pretrained backbone parameters are frozen.')
        for param in base_params:
            param.requires_grad = False

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    print('The newly initialized mlp layer uses {:.3f} x larger lr'.format(args.head_lr))
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)

    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))

    # only for preliminary test, formal exps should use fixed learning rate scheduler
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))

    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (a_input, v_input, labels) in enumerate(train_loader):

            B = a_input.size(0)
            a_input, v_input = a_input.to(device, non_blocking=True), v_input.to(device, non_blocking=True)
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
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')

        stats_complete, valid_loss_complete = validate(audio_model, test_loader, args, noise_to_audio=False, noise_to_vision=False)
        stats_audio, valid_loss_audio = validate(audio_model, test_loader, args, noise_to_audio=True, noise_to_vision=False)
        stats_vision, valid_loss_vision = validate(audio_model, test_loader, args, noise_to_audio=False, noise_to_vision=True)
        stats_audio_vision, valid_loss_audio_vision = validate(audio_model, test_loader, args, noise_to_audio=True, noise_to_vision=True)

        mAP = np.mean([stat['AP'] for stat in stats_complete])
        mAUC = np.mean([stat['auc'] for stat in stats_complete])
        acc = stats_complete[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy

        stats_list = [stats_complete, stats_audio, stats_vision, stats_audio_vision]
        valid_loss_list = [valid_loss_complete, valid_loss_audio, valid_loss_vision, valid_loss_audio_vision]
        config_names = ["No Noise", "Audio Noise Only", "Vision Noise Only", "Audio + Vision"]

        # 결과 출력
        print_validation_results(stats_list, valid_loss_list, config_names)

        print('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(acc)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

def validate(audio_model, val_loader, args, noise_to_audio=False, noise_to_vision=False, output_pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    
    # 노이즈 주입 활성화 여부를 args로 설정
    noise_params = {
        "noise_to_audio": noise_to_audio,
        "noise_to_vision": noise_to_vision,
    }
    
    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            # 배치 데이터를 디바이스로 이동
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)

            # 배치 단위 노이즈 주입
            if noise_params["noise_to_audio"] or noise_params["noise_to_vision"]:
                a_input, v_input = apply_noise_to_batch(a_input, v_input, noise_params)
            
            # 모델 예측
            with autocast():
                audio_output = audio_model(a_input, v_input, 'multimodal')

            # 결과 수집
            predictions = audio_output.to('cpu').detach()
            A_predictions.append(predictions)
            A_targets.append(labels.to('cpu'))

            # 손실 계산
            loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            # 배치 시간 업데이트
            batch_time.update(time.time() - end)
            end = time.time()

        # 전체 결과를 병합
        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)

        # 통계 계산
        stats = calculate_stats(audio_output, target)

    if output_pred == False:
        return stats, loss
    else:
        # multi-frame 평가를 위해 prediction과 target 반환
        return stats, audio_output, target

def save_images(batch_image, output_dir="output_images"):
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the batch and save each image
    for i in range(batch_image.size(0)):
        image = batch_image[i].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
        plt.imsave(os.path.join(output_dir, f"image_{i}.png"), image)
        
def print_validation_results(stats_list, valid_loss_list, config_names):
    print("===================== Validation Results =====================")
    print("\nConfiguration       |  Accuracy  |    mAP    |    AUC    |  d-prime  |  Valid Loss")
    print("---------------------------------------------------------------------------------")
    
    for stats, valid_loss, config in zip(stats_list, valid_loss_list, config_names):
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']  # 전체 정확도
        d_prime_value = d_prime(mAUC)
        
        print("{:<18} |  {:.6f}  |  {:.6f} |  {:.6f} |  {:.4f}  |  {:.6f}".format(
            config, acc, mAP, mAUC, d_prime_value, valid_loss
        ))
    
    print("==============================================================")
    print("\nValidation Completed!")