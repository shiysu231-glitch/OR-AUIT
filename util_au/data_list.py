import numpy as np
import random
from PIL import Image
import os
import pandas as pd
def make_dataset(image_list, au):
    len_ = len(image_list)
    images = [(image_list[i].strip(), au[i, :]) for i in range(len_)]

    return images


def pil_loader(prefix_path, path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Image value: [0,1]
    """自己加的代码"""
    full_path = os.path.join(prefix_path, path)
    with open(full_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(prefix_path, path):
    return pil_loader(prefix_path, path)


class ImageList(object):

    # def __init__(self, crop_size, path, phase='train', transform=None, target_transform=None,
    #              loader=default_loader):
    def __init__(self, suffix_path, prefix_path, transform=None, loader=default_loader):

        image_list = open(suffix_path + '_path.txt').readlines()

        au = np.loadtxt(suffix_path + '_AUintnorm.txt')
        ROI = np.loadtxt(suffix_path + '_ROI.txt')
        # ROI = np.loadtxt(r"D:\temporary_data\BP4D_part3_ROI_dlib.txt")

        imgs = make_dataset(image_list, au)
        if len(imgs) == 0:
            raise (RuntimeError('Found 0 images in subfolders of: ' + suffix_path + '\n'))

        self.prefix_path = prefix_path
        self.transform = transform
        self.loader = loader
        self.imgs = imgs
        # self.target_transform = target_transform
        # self.crop_size = crop_size
        # self.phase = phase

    def __getitem__(self, index):

        # path, land, biocular, au = self.imgs[index]
        path, au = self.imgs[index]
        img = self.loader(self.prefix_path, path)
        img = self.transform(img)
        return img, au

    def __len__(self):
        return len(self.imgs)

class ImageList3(object):
    def __init__(self, csv_path, prefix_path, transform=None, loader=default_loader):
        """
        csv_path: 新验证集的csv文件路径
        prefix_path: 图片的前缀路径（数据集根目录）
        """
        df = pd.read_csv(csv_path)

        # 取路径列
        image_list = df["image_dir"].tolist()
        # au = df.iloc[:, 2:].values
        au = df[["AU06", "AU10", "AU12", "AU14", "AU17"]].values.astype("float32") / 5.0
        # 取 AU 标签（去掉前两列 frame 和 image_dir）


        # 把路径和标签组合成 dataset
        imgs = list(zip(image_list, au))
        if len(imgs) == 0:
            raise RuntimeError(f'Found 0 images in {csv_path}')

        self.prefix_path = prefix_path
        self.transform = transform
        self.loader = loader
        self.imgs = imgs

    def __getitem__(self, index):
        path, au = self.imgs[index]
        img = self.loader(self.prefix_path, path)
        img = self.transform(img)
        return img, au

    def __len__(self):
        return len(self.imgs)
def make_dataset2(image_list, au, image_list2, au2, k):
    len_ = len(image_list)
    print("len(image_list):", len(image_list))
    print("len(image_list2):", len(image_list2))
    print("len(au):", len(au))
    print("len(au2):", len(au2))
    print("len(k):", len(k))
    print("len_:", len_)
    images = [(image_list[i].strip(), au[i,:], image_list2[i].strip(), au2[i,:], k[i]) for i in range(len_)]

    return images

class ImageList2(object):
    def __init__(self, suffix_path, prefix_path, suffix_path_1,transform=None, loader=default_loader):

        image_list = open(suffix_path + '_path.txt').readlines()
        au = np.loadtxt(suffix_path + '_AUintnorm.txt')
        image_list2 = open(suffix_path_1 + '_a.txt').readlines()
        au2 = np.loadtxt(suffix_path_1 + '_lab.txt')
        k = np.loadtxt(suffix_path_1 + '_b.txt')

        imgs = make_dataset2(image_list, au, image_list2, au2, k)
        if len(imgs) == 0:
            raise (RuntimeError('Found 0 images in subfolders of: ' + suffix_path + '\n'))

        self.prefix_path = prefix_path
        self.transform = transform
        self.loader = loader
        self.imgs = imgs

    def __getitem__(self, index):

        # path, land, biocular, au = self.imgs[index]
        path, au, path2, au2, k = self.imgs[index]
        img = self.loader(self.prefix_path, path)
        img = self.transform(img)
        img2 = self.loader(self.prefix_path, path2)
        img2 = self.transform(img2)
        return img, au, img2, au2, k

    def __len__(self):
        return len(self.imgs)