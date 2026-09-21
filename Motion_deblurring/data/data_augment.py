import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):
    """成对随机裁剪：对输入图和真值图用完全相同的裁剪参数（保证两者对齐）。"""

    def __call__(self, image, label):

        # 若设置了 padding 先对两图同时填充（继承自 torchvision.RandomCrop 的字段）
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # 需要时把宽/高补到目标尺寸（pad_if_needed）
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - label.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)  # 随机取一个裁剪位置

        # 用同一 (i,j,h,w) 裁剪两图，保证空间对齐
        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    """成对 Compose：把一组成对变换按顺序串起来，逐个应用到 (image, label)。"""

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    """成对随机水平翻转：以概率 p 同时翻转输入图和真值图。"""

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.
            label (PIL Image): label to be flipped.

        Returns:
            PIL Image,: randomly flipped image and label.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    """成对转张量：把 (PIL) 输入图和真值图都转为 Tensor（像素归一化到 [0,1]）。"""

    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            label (PIL Image or numpy.ndarray): label to be converted to tensor.

        Returns:
            Tensor: Converted image and label.
        """
        return F.to_tensor(pic), F.to_tensor(label)