# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util_au import data_list

"""添加的数据增强RandAugment()的dataset"""
import util_au.pre_process as pre


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # if is_train:
    #     transform = pre.image_train()
    # if not is_train:
    #     transform = pre.image_test()

    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    """
        dataset:
        返回的dataset都有以下三种属性
        self.classes：用一个 list 保存类别名称
        self.class_to_idx：类别对应的索引，与不做任何转换返回的 target 对应
        self.imgs：保存(img-path, class) tuple的 list
        ['cat', 'dog']
        {'cat': 0, 'dog': 1}
        [('./data/train\\cat\\1.jpg', 0), 
         ('./data/train\\cat\\2.jpg', 0), 
         ('./data/train\\dog\\1.jpg', 1), 
         ('./data/train\\dog\\2.jpg', 1)]
    """
    # dataset = datasets.ImageFolder(root, transform=transform)
    if is_train:
        dataset = data_list.ImageList2(suffix_path=args.train_suffix_path, prefix_path=args.train_prefix_path,suffix_path_1=args.train_suffix_path_1,
                                      transform=transform)
    else:
        if args.BP4D_test==1:  #仅用做BP4D的测试
            dataset = data_list.ImageList3(csv_path=args.test_csv_path, prefix_path=args.test_prefix_path,
                                      transform=transform)
        else:
            dataset = data_list.ImageList(suffix_path=args.val_suffix_path, prefix_path=args.val_prefix_path,
                                      transform=transform)

    print(dataset)

    return dataset

def build_dataset_1(is_train, args,fold_id=0):
    transform = build_transform(is_train, args)

    # if is_train:
    #     transform = pre.image_train()
    # if not is_train:
    #     transform = pre.image_test()

    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    """
        dataset:
        返回的dataset都有以下三种属性
        self.classes：用一个 list 保存类别名称
        self.class_to_idx：类别对应的索引，与不做任何转换返回的 target 对应
        self.imgs：保存(img-path, class) tuple的 list
        ['cat', 'dog']
        {'cat': 0, 'dog': 1}
        [('./data/train\\cat\\1.jpg', 0), 
         ('./data/train\\cat\\2.jpg', 0), 
         ('./data/train\\dog\\1.jpg', 1), 
         ('./data/train\\dog\\2.jpg', 1)]
    """
    # dataset = datasets.ImageFolder(root, transform=transform)
    if is_train:
        dataset = data_list.ImageList2(suffix_path=args.train_suffix_path, prefix_path=args.train_prefix_path,
                                      transform=transform)
    else:
        if args.BP4D_test==1:  #仅用做BP4D的测试
            dataset = data_list.ImageList3(csv_path=args.test_csv_path, prefix_path=args.test_prefix_path,
                                      transform=transform)
        else:
            if fold_id == 0:
                dataset = data_list.ImageList(
                    suffix_path=args.val_suffix_path,
                    prefix_path=args.val_prefix_path,
                    transform=transform)
            elif fold_id == 1:
                dataset = data_list.ImageList(
                    suffix_path=args.val_suffix_path_1,
                    prefix_path=args.val_prefix_path_1,
                    transform=transform)
            elif fold_id == 2:
                dataset = data_list.ImageList(
                    suffix_path=args.val_suffix_path_2,
                    prefix_path=args.val_prefix_path_2,
                    transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # pre_transform = transforms.Compose([
        #     transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
        #     transforms.CenterCrop(224),
        # ])
        # this should always dispatch to transforms_imagenet_train
        train_transform  = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        # return transforms.Compose([pre_transform, train_transform])
        return train_transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
