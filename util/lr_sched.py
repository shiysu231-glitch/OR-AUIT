# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
# def adjust_learning_rate(optimizer, progress, args):
#     """ progress = 当前epoch进度(含iter)，范围[0, total_epoch] """
#     # 学习率调度系数 (cosine 或 warmup)
#     if progress < args.warmup_epochs:
#         factor = progress / args.warmup_epochs
#     else:
#         factor = 0.5 * (1. + math.cos(math.pi * (progress - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
#
#     # 这里不再用全局 lr，而是按 param_group 初始 lr 缩放
#     for param_group in optimizer.param_groups:
#         base_lr = param_group.get("initial_lr", param_group["lr"])
#         param_group["lr"] = base_lr * factor
