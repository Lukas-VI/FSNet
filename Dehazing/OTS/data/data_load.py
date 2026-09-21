import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # 允许加载截断/损坏的图片，避免 OTS 大数据集个别坏图中断训练

def train_dataloader(path, batch_size=64, num_workers=0):
    """构造训练加载器（OTS：<path>/train/，训练时在 Dataset 内做随机裁剪 ps=256 增强）。"""
    image_dir = os.path.join(path, 'train')

    dataloader = DataLoader(
        DeblurDataset(image_dir, ps=256),   # 训练时裁 256x256
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    """构造测试加载器（OTS：<path>/test/，返回整图并带文件名）。"""
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    """构造验证加载器（复用测试目录，is_valid=True 表示读取 .png 真值）。"""
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

import random
class DeblurDataset(Dataset):
    """成对的（hazy/退化, gt/真值）数据集（OTS 版本）。

    关键差异：训练时会加载整图再随机裁剪 ps 大小；真值文件名匹配规则为
    『输入名按 _ 分割取第 0 段 + .png(train 用 .jpg)』。
    """
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid
        self.ps = ps   # 训练裁剪尺寸

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx])).convert('RGB')
        # 验证/测试时 GT 为 .png；训练时 GT 为 .jpg
        if self.is_valid or self.is_test:
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png')).convert('RGB')
        else:
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.jpg')).convert('RGB')
        ps = self.ps

        if self.ps is not None:
            # ---- 训练增强：随机裁剪 ps 大小 + 随机水平翻转 ----
            image = F.to_tensor(image)
            label = F.to_tensor(label)

            hh, ww = label.shape[1], label.shape[2]  # 高/宽

            rr = random.randint(0, hh-ps)   # 随机裁剪起点
            cc = random.randint(0, ww-ps)

            image = image[:, rr:rr+ps, cc:cc+ps]   # 裁剪同一区域保证对齐
            label = label[:, rr:rr+ps, cc:cc+ps]

            if random.random() < 0.5:   # 50% 概率水平翻转（image 与 label 一起翻）
                image = image.flip(2)
                label = label.flip(2)
        else:
            # 无 ps（测试/验证）直接转张量
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            name = self.image_list[idx]
            return image, label, name   # 测试额外返回文件名
        return image, label



    @staticmethod
    def _check_image(lst):
        # 校验文件为常见图片格式
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError