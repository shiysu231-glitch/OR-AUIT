# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import torch.nn.functional as F
import torch
from util_au.tt import ICC
# from timm.data import Mixup
from timm.utils import accuracy
"""用自己的Mixup"""
from util_au.mixup import Mixup

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F
# 引入numpy
import numpy as np
# 引入accuracy_score, f1_score
from sklearn.metrics import accuracy_score, f1_score

from util_au.eva import calMAE,calcICC
from util_au.lr_set import vit_param_groups
import torch
import torch.nn as nn
#自定义损失函数
class HingeLoss_A(nn.Module):
    def forward(self, x, y, target):
        loss = torch.max(torch.zeros_like(x),  - (x - y)) * target
        return loss.mean()
akw12 = [0.09384185357042017, 0.14565932525998207, 0.030243717576599334, 0.1827265123036339, 0.0352218538414212, 0.1263645544107405, 0.020438556095412376, 0.06691263537174282, 0.05413658373503145, 0.20707736454191097, 0.013355568819372539, 0.02402147447373255]
akw12 = torch.tensor(akw12).cuda()
class HingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        loss = (torch.max(torch.zeros_like(output1), self.margin - (output1 - output2))) * target
        return loss.mean()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    k = 0
    for data_iter_step, (samples, targets, samples2, targets2, id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets = targets.float()
        samples2 = samples2.to(device, non_blocking=True)
        targets2 = targets2.to(device, non_blocking=True)
        targets2 = targets2.float()
        id = id.to(device, non_blocking = True)

        with torch.cuda.amp.autocast():
            outputs, fc1, outputs2, fc2 = model(samples, samples2, k)
            outputs = outputs.squeeze()
            outputs2 = outputs2.squeeze()
            #mse_loss

            loss1 = F.mse_loss(outputs, targets)
            loss2 = F.mse_loss(outputs2, targets2)
            #hingeloss
            hinge = HingeLoss(margin=args.margin)
            loss_p = hinge(fc1, fc2, id)
            loss = (loss2 + loss1) + loss_p*args.factor
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
#end
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, device,args,threshold):
# def evaluate(data_loader, model, device,args):

    criterion = torch.nn.L1Loss()
    # 记录样本总数
    all_batchsize = 0

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    k = 1

    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, images, k)
            output = output.squeeze(1)

            loss = criterion(output, target)
        au_output = output

        if i == 0:
            all_output = au_output.data.cpu().float()
            all_au = target.data.cpu().float()
        else:
            all_output = torch.cat((all_output, au_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, target.data.cpu().float()), 0)

        batch_size = images.shape[0]
        all_batchsize = all_batchsize + batch_size
        metric_logger.update(loss=loss.item())

    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_actual = all_au.data.numpy()



    AUoccur_actual = AUoccur_actual.transpose()     #real
    AUoccur_pred = AUoccur_pred_prob.transpose()    # predict

    AUoccur_pred[AUoccur_pred < threshold] = 0.0
    AUoccur_pred[AUoccur_pred >1] = 1
    AUoccur_pred *= 5
    AUoccur_actual *= 5

    iccv = np.zeros(AUoccur_actual.shape[0])
    maev = np.zeros(AUoccur_actual.shape[0])

    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]
        ICCa = []
        ICCa.append(curr_pred)
        ICCa.append(curr_actual)
        ICCa = np.transpose(ICCa)
        iccv[i] = ICC(3,'single',ICCa)
        maev[i] = calMAE(curr_pred,curr_actual)
    #DIFSA
    sumicc = sum(iccv)
    iccmean = sumicc / args.nb_classes
    summae = sum(maev)
    maemean = summae / args.nb_classes


    print('icc')
    print(iccv)
    print('meanICC')
    print(iccmean)
    print('mae')
    print(maev)
    print('meanMAE')
    print(maemean)

    return iccmean,maemean,iccv,maev

@torch.no_grad()
# def evaluate_mse(data_loader, model, device, args):
def evaluate_mse(data_loader, model, device, args,threshold):

    criterion_mae = torch.nn.L1Loss()
    criterion_mse = torch.nn.MSELoss()

    all_batchsize = 0
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    k = 1

    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images, images, k)
            output = output.squeeze(1)
            loss = criterion_mae(output, target)  # 默认用MAE更新loss（可改成MSE）

        au_output = output

        if i == 0:
            all_output = au_output.data.cpu().float()
            all_au = target.data.cpu().float()
        else:
            all_output = torch.cat((all_output, au_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, target.data.cpu().float()), 0)

        batch_size = images.shape[0]
        all_batchsize += batch_size
        metric_logger.update(loss=loss.item())

    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_actual = all_au.data.numpy()

    AUoccur_actual = AUoccur_actual.transpose()
    AUoccur_pred = AUoccur_pred_prob.transpose()

    AUoccur_pred[AUoccur_pred < 0.0] = 0.0
    AUoccur_pred *= 5
    AUoccur_actual *= 5

    iccv = np.zeros(AUoccur_actual.shape[0])
    maev = np.zeros(AUoccur_actual.shape[0])
    msev = np.zeros(AUoccur_actual.shape[0])

    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]

        ICCa = np.transpose([curr_pred, curr_actual])
        iccv[i] = ICC(3, 'single', ICCa)

        maev[i] = np.mean(np.abs(curr_pred - curr_actual))      # MAE
        msev[i] = np.mean((curr_pred - curr_actual) ** 2)       # MSE

    iccmean = np.mean(iccv)
    maemean = np.mean(maev)
    msemean = np.mean(msev)

    print('icc:', iccv)
    print('meanICC:', iccmean)
    print('mae:', maev)
    print('meanMAE:', maemean)
    print('mse:', msev)
    print('meanMSE:', msemean)

    return iccmean, maemean, msemean, iccv, maev, msev
