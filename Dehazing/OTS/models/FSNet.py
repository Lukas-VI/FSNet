import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

# ============================================================
# FSNet.py —— 网络主体结构
# 编码器-解码器(Encoder-Decoder)风格的多尺度网络，输出多个分辨率的复原图。
# 处理流程：多输入分辨率特征 -> FAM(特征融合) + SCM(从输入图像本身提取条件特征)
# 进行融合 -> 逐级下采样编码 -> 逐级上采样解码 -> 输出各级复原结果。
# 说明：该模型在 Dehazing / Desnowing / Motion_deblurring / OTS 等任务中是同一套
# 结构（各任务副本仅存在空行/命名差异，逻辑一致）。
# ============================================================

class EBlock(nn.Module):
    """Encoder 残差块：串联 num_res 个 ResBlock，最后一个开启动态滤波。"""
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        # 前 num_res-1 个普通残差块，最后 1 个 filter=True（启用动态滤波）。
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    """Decoder 残差块：结构与 EBlock 完全一致（对称的左右手）。"""
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EBlock1(nn.Module):
    """最内层 Encoder 块：内部使用轻量 Unet（含一步降/升采样），
    用于最低分辨率层的深层特征提取。"""
    def __init__(self, out_channel, num_res=8):
        super(EBlock1, self).__init__()

        self.layers = Unet(out_channel, out_channel, num_res)
    def forward(self, x):
        return self.layers(x)


class DBlock1(nn.Module):
    """最内层 Decoder 块：与 EBlock1 对称，同样用 Unet 上采样/恢复。"""
    def __init__(self, channel, num_res=8):
        super(DBlock1, self).__init__()

        self.layers = Unet(channel, channel, num_res)
    def forward(self, x):
        return self.layers(x)

class SCM(nn.Module):
    """Spatial Co-Module（空间协同模块），FCN 类网络常用的多尺度引导分支。

    作用：从输入的退化图像(3 通道 RGB)本身提取出一组"条件特征"，
    后续通过 FAM 与主分支特征融合，充当先验/引导信息。类似 NAFNet 的 SCM。
    """
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        # 一系列 1x1 / 3x3 卷积堆叠，逐步把通道从 3 扩到 out_plane，最后过 InstanceNorm。
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)  # 实例归一化，抑制风格/光照变化
        )

    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    """Feature Attention Module（特征融合/注意力模块）。

    输入两路特征 x1、x2，沿通道拼接后用一层 3x3 卷积融合到 channel 维。
    简单起见此处仅做通道拼接+卷积（轻量版特征融合）。
    """
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class FSNet(nn.Module):
    """FSNet 主网络。

    整体是下采样(Encoder)/上采样(Decoder)的多尺度结构，有三个分辨率尺度
    (256->128->64)。输出 outputs 是三张不同分辨率的复原图（列表，下标0/1/2
    分别对应 1/4、1/2、原尺寸），training 时与对应分辨率 GT 计算损失。
    """
    def __init__(self, num_res=16):
        super(FSNet, self).__init__()

        base_channel = 32  # 基础通道数，后续按下采样逐级翻倍

        # ---- 编码器（自顶向下，逐级提取特征到更低分辨率）----
        self.Encoder = nn.ModuleList([
            EBlock1(base_channel, num_res),              # 256 分辨率，最内层用 Unet
            EBlock(base_channel*2, num_res),             # 128 分辨率
            EBlock(base_channel*4, num_res),             # 64 分辨率，通道最多
        ])

        # ---- 分辨率/通道转换卷积组（负责上下采样的卷积）----
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),                 # [0] 输入RGB->32 通道
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),     # [1] 32->64，下采样
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),   # [2] 64->128，下采样
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),  # [3] 128->64，转置卷积上采样
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),   # [4] 64->32，转置卷积上采样
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)                 # [5] 32->3，输出原分辨率复原图
        ])

        # ---- 解码器（自底向上，逐级上采样并恢复）----
        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),   # 64 分辨率
            DBlock(base_channel * 2, num_res),   # 128 分辨率
            DBlock1(base_channel, num_res)       # 256 分辨率，最内层用 Unet
        ])

        # ---- 解码时融合 skip 连接的 1x1 卷积组 ----
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        # ---- 中间两个尺度的输出头（把 128/64 通道压回 3 通道 RGB）----
        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),  # 64 分辨率输出
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),  # 128 分辨率输出
            ]
        )

        # ---- 各层的特征融合(FAM)与引导(SCM)模块 ----
        self.FAM1 = FAM(base_channel * 4)   # 用于 64 分辨率
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)   # 用于 128 分辨率
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        # 把输入 x(256) 逐级缩小到 1/2(128) 和 1/4(64)，作为 SCM 引导的输入。
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)   # 128 分辨率引导特征
        z4 = self.SCM1(x_4)   # 64 分辨率引导特征

        outputs = list()
        # ===== 编码路径（Encoder）=====
        # 256 分辨率
        x_ = self.feat_extract[0](x)          # RGB->32ch，保持 256
        res1 = self.Encoder[0](x_)            # 保存残差（用于后面 skip）
        # 128 分辨率
        z = self.feat_extract[1](res1)        # 32->64ch，下采样到 128
        z = self.FAM2(z, z2)                  # 与 SCM2 引导特征融合
        res2 = self.Encoder[1](z)             # 保存残差
        # 64 分辨率
        z = self.feat_extract[2](res2)        # 64->128ch，下采样到 64
        z = self.FAM1(z, z4)                  # 与 SCM1 引导特征融合
        z = self.Encoder[2](z)

        # ===== 解码路径（Decoder），自底向上 =====
        z = self.Decoder[0](z)                # 64 分辨率解码
        z_ = self.ConvsOut[0](z)              # 输出 64 分辨率复原图（1/4 尺寸）
        # 128 分辨率
        z = self.feat_extract[3](z)           # 上采样到 128
        outputs.append(z_+x_4)                # 与缩小版输入相加（残差式），得到 1/4 尺寸输出

        z = torch.cat([z, res2], dim=1)       # skip 拼接编码器 res2
        z = self.Convs[0](z)                  # 1x1 融合回 64ch
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)              # 128 分辨率输出（1/2 尺寸）
        # 256 分辨率
        z = self.feat_extract[4](z)           # 上采样到 256
        outputs.append(z_+x_2)                # 残差式，得到 1/2 尺寸输出

        z = torch.cat([z, res1], dim=1)       # skip 拼接编码器 res1
        z = self.Convs[1](z)                  # 融合回 32ch
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)           # 32->3ch，输出原分辨率复原图
        outputs.append(z+x)                   # 残差式与原始输入相加

        return outputs  # 返回 [1/4尺寸图, 1/2尺寸图, 原尺寸图]


def build_net():
    """便捷构造函数：实例化一个默认参数的 FSNet。"""
    return FSNet()