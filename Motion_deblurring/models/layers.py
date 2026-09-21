import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
train_size = (1,3,256,256)  # 训练输入尺寸，用于把池化核/步长按输入大小换算（自适应缩放）

# ============================================================
# FSNet 的基础网络层模块（Motion Deblurring / GoPro 版本）
# 结构上属于"长版"(含 AvgPool2d)，与 OTS 版本基本一致，唯一差异是
# 各处 AvgPool2d 的 base_size 取较小值 80（OTS 为 210）。
# 其余 BasicConv / ResBlock / Unet / dynamic_filter 逻辑与 ITS 版本一致。
# ============================================================

class AvgPool2d(nn.Module):
    """可调均值池化（按输入分辨率动态决定池化核/步长）。

    目的：训练输入是 256x256 时，池化核取 base_size；推理时输入可能是任意分辨率
    （更大或更小），按比例缩放 kernel_size 与步长，保证"池化感受野"在不同
    分辨率下保持一致（即使得池化结果的相对语义对齐）。

    fast_imp=True 时用积分图(cumsum)近似实现求均值池化，速度更快但非完全等价。
    """
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size   # 基准池化核（相对 256 训练尺寸）
        self.auto_pad = auto_pad     # 是否要把输出补回与输入同分辨率

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5,4,3,2,1]       # 快速近似时的下采样步长候选（用于可整除缩采样）
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        # ---- 根据输入尺寸动态计算池化核大小 ----
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            # 按输入/训练尺寸比例缩放核大小
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//train_size[-2]
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//train_size[-1])

        if self.fast_imp:   # Non-equivalent implementation but faster  —— 快速近似分支
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:
                out = F.adaptive_avg_pool2d(x,1)   # 核比图还大，直接全局池化
            else:
                # 选择可整除输入尺寸的缩放步长，并受 max_r 限制
                r1 = [r for r in self.rs if h%r==0][0]
                r2 = [r for r in self.rs if w%r==0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                # 先隔点采样再用积分图(cumsum)做块求和求平均 —— 相对原始 kernel 的快速近似
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))  # 缩放回原尺寸
        else:
            # ---- 精确分支：直接用积分图求滑动窗均值 ----
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)   # 二维积分图（前缀和）
            s = torch.nn.functional.pad(s, (1,0,1,0)) # pad 0 for convenience —— 便于边界计算
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])

            # 用积分图四角差分算出每个窗内之和（经典池化加速技巧）
            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3
            out = out / (k1*k2)   # 求平均

        # ---- 把输出补回与输入同分辨率（replicate 复制边缘）----
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


class BasicConv(nn.Module):
    """基础的卷积封装层（与 ITS 版本相同）。

    将『卷积/转置卷积 + 可选BN + GELU激活』包装为 nn.Sequential 复用。
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        # 有 BN 时关掉卷积 bias（BN 自带偏置，避免冗余）
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())   # 激活函数选用 GELU
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Gap(nn.Module):
    """全局均值池化后的低频/高频分离并加权重建（OTS 用 AvgPool2d 实现池化）。"""
    def __init__(self, in_channel) -> None:
        super().__init__()

        # 每个通道一个可学习缩放系数，初始为 0
        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = AvgPool2d(base_size=80)  # 相对 256 训练尺寸的基准池化核为 80（Motion 版本较小）

    def forward(self, x):
        x_d = self.gap(x)  # 全局低频（通道均值）
        # 高频 = 原图 - 低频，用 (fscale_h+1) 缩放（初始=恒等）
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        # 低频本身用 fscale_d 独立缩放（初始为 0 即去除低频）
        x_d = x_d  * self.fscale_d[None, :, None, None]
        return x_d + x_h


class ResBlock(nn.Module):
    """残差块，FSNet 核心重复单元（与 ITS 版本一致；filter=True 时启用动态滤波）。"""
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter

        # 动态滤波：通道前半部分 3x3、后半部分 5x5（多尺度）
        self.dyna = dynamic_filter(in_channel//2) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel//2, kernel_size=5) if filter else nn.Identity()

        # 通道后半部分两路：局部 patch 高低频分离 + 全局高低频分离
        self.localap = Patch_ap(in_channel//2, patch_size=2)
        self.global_ap = Gap(in_channel//2)


    def forward(self, x):
        out = self.conv1(x)

        if self.filter:
            # 沿通道对半分割，分别做 3x3 / 5x5 动态滤波后再次拼接
            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)

        # 再按通道对半分：一半全局高低频分离，一半局部 patch 高低频分离
        non_local, local = torch.chunk(out, 2, dim=1)
        non_local = self.global_ap(non_local)
        local = self.localap(local)
        out = torch.cat((non_local, local), dim=1)
        out = self.conv2(out)
        return out + x   # 残差连接

class Unet(nn.Module):
    """最内层轻量 U 型结构（与 ITS 一致）：num_res 个 ResBlock，中间做一次降/升采样。"""
    def __init__(self, in_channel, out_channel, num_res):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_res-1):
            self.layers.append(ResBlock(in_channel, out_channel))
        self.layers.append(ResBlock(in_channel, out_channel, filter=True))  # 最后一个开动态滤波

        # 内部 2x2 depth-wise 下采样
        self.down = nn.Conv2d(in_channel, in_channel, kernel_size=2, stride=2, groups=in_channel)
        self.num_res = num_res

        # 1x1 卷积用于上采样后融合 skip
        self.conv = nn.Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1)
    def forward(self, x):
        res = x.clone()

        for i, layer in enumerate(self.layers):
            # 约 1/4 处下采样并保存 skip
            if i == self.num_res//4:
                skip = x
                x = self.down(x)
            # 约 3/4 处上采样回输入尺寸并融合 skip
            if i == self.num_res - self.num_res//4:
                x = F.upsample(x, res.shape[2:], mode='bilinear')
                x = self.conv(torch.cat((x, skip), dim=1))
            x = layer(x)

        return x + res

class dynamic_filter(nn.Module):
    """动态滤波器（与 ITS 一致）：由输入内容生成滤波核，自适应分离低频成分。"""
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        # 生成权重：输出 group*k^2 通道
        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)   # 沿核内元素 softmax，权重和=1
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size//2)  # 反射 padding

        self.ap = nn.AdaptiveAvgPool2d((1, 1))   # 全局池化生成特征

        self.modulate = SFconv(inchannels)   # 高低频选择性融合

    def forward(self, x):
        identity_input = x # 本例输入形状示例：3,32,64,64

        # 生成滤波核
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)

        # im2col 展开成滑动窗，并按 group 分组
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)

        low_filter = self.act(low_filter)   # 核内 softmax 归一

        # 加权求和得到局部低频
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part   # 高频 = 原图 - 低频
        out = self.modulate(low_part, out_high)  # 高低频融合
        return out

class SFconv(nn.Module):
    """高低频选择性融合（与 ITS 兼容，但池化改用 AvgPool2d）。"""
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features/r), L)   # bottleneck 维度（压缩率 r，下限 L）
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)   # 压缩
        self.fcs = nn.ModuleList([])                 # M 个上采样分支（M=2）
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = AvgPool2d(base_size=80)          # 用可调池化（Motion 版本 base_size=80）

        self.out = nn.Conv2d(features, features, 1, 1, 0)
    def forward(self, low, high):
        emerge = low + high         # 聚合
        emerge = self.gap(emerge)   # 全局池化获得通道描述子

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)   # 高/低频注意力
        low_att = self.fcs[1](fea_z)

        attention_vectors = torch.cat([high_att, low_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)  # 两者权重和为 1
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)

        fea_high = high * high_att
        fea_low = low * low_att

        out = self.out(fea_high + fea_low)
        return out

class Patch_ap(nn.Module):
    """局部 patch 高低频分离（与 ITS 兼容，但池化改用 AvgPool2d）。"""
    def __init__(self, inchannel, patch_size):
        super(Patch_ap, self).__init__()

        # self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.ap = AvgPool2d(base_size=80)   # 可调池化（Motion 版本 base_size=80）

        self.patch_size = patch_size
        self.channel = inchannel * patch_size**2
        self.h = nn.Parameter(torch.zeros(self.channel))   # 高频系数
        self.l = nn.Parameter(torch.zeros(self.channel))   # 低频系数

    def forward(self, x):

        # 用 einops 把空间重排为 patch 网格
        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        # 把每个 patch 摊平到通道维
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        low = self.ap(patch_x)   # 每个 patch 的局部低频
        high = (patch_x - low) * self.h[None, :, None, None]   # 高频残差 * 系数
        out = high + low * self.l[None, :, None, None]         # 重建
        # 还原空间排布
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)

        return out