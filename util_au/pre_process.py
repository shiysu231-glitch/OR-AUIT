import numpy as np
from torchvision import transforms
from PIL import Image
import random
from RandAugment import RandAugment
"""Transform"""


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

# def image_train():
#     normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                      std=[0.5, 0.5, 0.5])
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         normalize
#     ])
#     return transform.transforms.insert(0, RandAugment(9, 0.5))


# def image_test():
#     normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                      std=[0.5, 0.5, 0.5])
#     return transforms.Compose([
#         transforms.ToTensor(),
#         normalize
#     ])


class image_train(object):

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        transform.transforms.insert(0, RandAugment(9, 0.5))
        img = transform(img)
        return img

class image_test(object):

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img
