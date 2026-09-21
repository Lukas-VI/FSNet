import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    """构造训练数据加载器（数据集位于 <path>/train/，内部含 hazy/ 与 gt/ 两个子目录）。"""
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        # 训练时做联合数据增强：随机裁剪 256 + 随机水平翻转 + 转张量（成对作用于 input/label）
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
        shuffle=True,          # 训练打乱
        num_workers=num_workers,
        pin_memory=True        # 锁页内存加速 CPU->GPU 拷贝
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    """构造测试数据加载器（数据集位于 <path>/test/）。"""
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),  # is_test=True 时返回图像名
        batch_size=batch_size,
        shuffle=False,          # 测试不打乱，保证顺序可复现
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    """构造验证数据加载器（复用测试目录 <path>/test/）。"""
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    """成对的（退化图, 清晰图）图像数据集。

    目录约定：
      <image_dir>/hazy/  —— 退化（有雾）输入图
      <image_dir>/gt/    —— 清晰真值图（GT）。
    注意：GT 文件名 = 输入文件名按 '_' 分割取第 0 段 + '.png'，
    即要求输入与 GT 有公共前缀命名。
    """
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)  # 校验文件都是图片格式
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 输入图：hazy 目录下同名文件
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx]))
        # 真值图：gt 目录下按命名匹配（前缀 + .png）
        label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png'))

        # 有 transform 走成对增强，否则直接转张量
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name  # 测试时额外返回文件名，便于保存结果
        return image, label

    @staticmethod
    def _check_image(lst):
        # 简单校验：文件后缀必须是常见图片格式，否则抛异常
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError