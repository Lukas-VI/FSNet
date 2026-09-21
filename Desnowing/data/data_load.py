"""Desnowing 数据集加载（去除运动模糊任务沿用同套代码，目录约定有差异）。

目录约定（Desnowing）：
  <image_dir>/Snow/  —— 有雪退化输入图
  <image_dir>/Gt/    —— 清晰真值图（GT），与输入同名。
"""
import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    """构造训练加载器（Desnowing：<path>/train2500/），训练时做成对增强。"""
    image_dir = os.path.join(path, 'train2500')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    """构造测试加载器（Desnowing：<path>/test2000/）。"""
    image_dir = os.path.join(path, 'test2000')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    """构造验证加载器（复用测试目录 test2000）。"""
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test2000')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    """成对的（Snow/有雪, Gt/清晰真值）数据集（Desnowing 版本）。"""
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'Snow/'))
#        self._check_image(self.image_list)   # 原实现注释掉了格式校验（保留原注释）
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 输入图：Snow 目录下同名
        image = Image.open(os.path.join(self.image_dir, 'Snow', self.image_list[idx]))
        # label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx].split('.')[0]+'.jpg'))#srrs+jpg
        # 上述曾用命名匹配方式读取 GT（# srrs+jpg），当前直接按同名读取
        label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name  # 测试额外返回文件名
        return image, label