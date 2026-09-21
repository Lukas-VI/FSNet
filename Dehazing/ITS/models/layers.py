import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ============================================================
# FSNet 的基础网络层模块（layers）
# 这是整个模型最底层的"积木"，被 FSNet.py 中的各 Block 调用。
# 整体思想：用 Residual 结构 + 动态滤波(Dynamic Filter) + 高低频分离来恢复
# 被退化(雾/雨/雪/模糊)破坏的图像细节。
# ============================================================

class BasicConv(nn.Module):
    """基础的卷积封装层。

    把『卷积/转置卷积 + 可选BN + GELU激活』包装成一个 nn.Sequential，
    方便上层代码用一行构造出常见的卷积块。名字里的 Basic 表示最普通的卷积单元。
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        # 顺序：norm(BN) 在前、后面接 relu。注意：若 norm=True 则把 bias 关掉，
        # 因为 BN 本身自带偏置项，卷积再带 bias 会冗余（这也是 PyTorch 的标准做法）。
        if bias and norm:
            bias = False

        # 普通卷积用 SAME 式 padding（kernel_size//2 保持尺寸不变）。
        # PyTorch 的 Conv 默认 padding=0，这里手动补到"输出尺寸=输入尺寸/stride"。
        padding = kernel_size // 2
        layers = list()
        if transpose:
            # 转置卷积（上采样）的 padding 公式与普通卷积不同，减 1 是几何上的对称需求。
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            # 激活用 GELU 而不是 ReLU，GELU 更平滑，近年视觉网络（含 Transformer 系）常用。
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Gap(nn.Module):
    """全局平均池化(GAP)后做低频/高频分离并加权重建。

    思路：x_d 是整张图信息的"低频"（全局均值），x_h 是"高频"（减掉均值后的残差）。
    用两个可学习缩放系数 fscale_d/fscale_h 分别调制低频和高频的贡献，
    让网络自己学会该强调哪部分频率信息。
    """
    def __init__(self, in_channel) -> None:
        super().__init__()

        # 每个通道一个尺度参数，初始为 0（即初始不改变原特征）。
        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        # 全局平均池化到 1x1，得到每个通道的全局均值。
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        # x_d: 全局低频（每个通道一个常数）
        x_d = self.gap(x)
        # x_h: 残差（原图-低频），用 (fscale_h+1) 缩放。+1 让初始状态等价于恒等(系数为1)。
        # 下面的 None 用于扩维：None,:,None,None 将 shape=[C] 扩展为 [1,C,1,1] 以便广播。
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        # 低频本身也用 fscale_d 独立缩放（初始为0 表示减掉全部低频，即只保留高频残差）。
        x_d = x_d  * self.fscale_d[None, :, None, None]
        return x_d + x_h


class ResBlock(nn.Module):
    """残差块，FSNet 中最核心的重复单元。

    结构：conv1 -> (可选)动态滤波 -> 通道切分+高低频分别处理 -> concat -> conv2 -> +x 残差。
    说明：
      - filter=True 时启用动态滤波分支（对通道的前半部分做 3x3 动态卷积，后半做 5x5）。
      - 最后的余项分成两半：一半走全局高低频分离(Gap，善于捕捉全局结构)，
        另一半走局部 patch 的高低频分离(Patch_ap，善于捕捉局部纹理)。
    """
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter  # 是否启用动态滤波分支

        # 动态滤波：针对通道的前/后半部分分别做，尺寸 3x3 与 5x5（多尺度感受野）。
        # 不用 filter 时退化为恒等(Identity)，不参与计算。
        self.dyna = dynamic_filter(in_channel//2) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel//2, kernel_size=5) if filter else nn.Identity()

        # 通道后半部分拆两路：局部(patch)优先的高频分离 / 全局高低频分离。
        self.localap = Patch_ap(in_channel//2, patch_size=2)
        self.global_ap = Gap(in_channel//2)


    def forward(self, x):
        out = self.conv1(x)

        # ---- 动态滤波分支 ----
        if self.filter:
            # 沿通道维对半分：k3 走 3x3 动态滤波，k5 走 5x5 动态滤波。
            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)

        # ---- 高低频并行处理 ----
        # 再次按通道对半分：一半做全局高低频分离，一半做局部 patch 高低频分离。
        non_local, local = torch.chunk(out, 2, dim=1)
        non_local = self.global_ap(non_local)
        local = self.localap(local)
        out = torch.cat((non_local, local), dim=1)
        out = self.conv2(out)
        return out + x  # 残差连接，让网络只学"增量/残差"，利于梯度传递与更深训练

class Unet(nn.Module):
    """嵌入到 Encoder/Decoder 最内层的轻量 U 型网络。

    作用：对最底层(特征分辨率最低、通道数最多)的特征再做一次"降采样->恢复"的
    小 U 型结构，扩大感受野并增强特征表达。与标准 U-Net 区别：
    num_res 个 ResBlock 串联，在中间某处做一次 2x2 depth-wise 降采样再上采样回来。
    """
    def __init__(self, in_channel, out_channel, num_res):
        super().__init__()

        # 前面 num_res-1 个是普通残差块，最后 1 个开启动态滤波(filter=True)。
        self.layers = nn.ModuleList()
        for i in range(num_res-1):
            self.layers.append(ResBlock(in_channel, out_channel))
        self.layers.append(ResBlock(in_channel, out_channel, filter=True))

        # analog 内部降采样：2x2、步长2、group=channel，即对每个通道独立下采样（depth-wise）。
        self.down = nn.Conv2d(in_channel, in_channel, kernel_size=2, stride=2, groups=in_channel)
        self.num_res = num_res

        # 上采样后把两路特征（下采样前的 skip + 当前）叠加融合到 in_channel 维。
        self.conv = nn.Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1)
    def forward(self, x):
        res = x.clone()  # 保存最外层残差（用于最后的 +res）。

        for i, layer in enumerate(self.layers):
            # 大约串到 1/4 处时进行一次下采样，并保存 skip 特征供后面上采样用。
            if i == self.num_res//4:
                skip = x
                x = self.down(x)
            # 大约串到 3/4 处时把特征上采样回输入尺寸，并用 1x1 conv 融合 skip。
            if i == self.num_res - self.num_res//4:
                x = F.upsample(x, res.shape[2:], mode='bilinear')  # 上采样到与最外层输入同尺寸
                x = self.conv(torch.cat((x, skip), dim=1))
            x = layer(x)  # 逐个经过残差块

        return x + res  # 最外层残差连接

class dynamic_filter(nn.Module):
    """动态滤波器（Dynamic Filter）——可学习的自适应滤波核。

    核心思想：用全局池化+1x1卷积动态地"预测"一组滤波权重，再去对图像做加权
    局部求和（im2col 展开 + 逐元素加权），从而自适应地区分/分离出"低频"成分。
    这与普通固定卷积核的区别在于：滤波核是由输入内容决定的（dynamic）。
    """
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group  # 把通道分组，每组生成一组权重，减少参数量

        # 由全局特征生成滤波权重：输出 group*kernel_size^2 个通道（每个通道一组滤波核）。
        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)  # 沿核内各元素做 softmax，保证权重和为 1（类似加权平均）
        # kaiming 初始化权重，配合 relu 的 fan_out 模式。
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # 两个可学习尺度参数（低频/高频权重），见 forward。
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        # 反射式 padding（边界镜像），用于保持卷积后尺寸。
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        # 全局平均池化用于生成权重特征。
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        # SFconv 用于低频/高频特征的选择性融合。
        self.modulate = SFconv(inchannels)

    def forward(self, x):
        identity_input = x # 本例输入形状示例：3,32,64,64

        # ---- 生成动态滤波核 ----
        low_filter = self.ap(x)   # 全局池化，提取每个通道的整体统计 -> [n, c, 1, 1]
        low_filter = self.conv(low_filter)   # 1x1 卷积生成滤波权重
        low_filter = self.bn(low_filter)     # 归一化

        # ---- im2col：把图像切成 kernel_size x kernel_size 的块 ----
        n, c, h, w = x.shape
        # unfold 取出所有滑动窗口；reshape 成 [n, group, c//group, k^2, h*w]，
        # 便于每个分组独立做加权。
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        # 把滤波权重 reshape 成与上面一致的形状并加 softmax（保证权值归一）。
        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)

        low_filter = self.act(low_filter)

        # 沿核维度加权求和，得到"局部低频"部分 low_part。
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        # 高频 = 原图 - 低频，再交给 SFconv 做高低频的可学习融合。
        out_high = identity_input - low_part
        out = self.modulate(low_part, out_high)
        return out


class SFconv(nn.Module):
    """Selective Fusion Conv：对高频/低频特征做自适应加权融合（类似 SE 注意力）。

    流程：high+low 池化 -> 压缩(fc) -> 两组上采样(high/low 各得一份注意力权重) ->
    softmax 保证两者权重和为 1 -> 加权合并 -> 1x1 输出。
    相当于让网络学会"这一处该侧重低频还是高频"。
    """
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features/r), L)  # bottleneck 通道数（压缩率 r，且不低于 L）
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)      # 压缩：features -> d
        self.fcs = nn.ModuleList([])                    # 两个上采样分支
        for i in range(M):                              # M=2：一个用于 high，一个用于 low
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)              # 全局池化得到聚合特征

        self.out = nn.Conv2d(features, features, 1, 1, 0)

    def forward(self, low, high):
        emerge = low + high          # 高低频相加，得到聚合信息
        emerge = self.gap(emerge)    # 全局池化 -> 通道级描述子 [n, c, 1, 1]

        fea_z = self.fc(emerge)      # 压缩

        # 分别生成高/低频的注意力权重
        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)

        # 拼接后 softmax，让 (high_att, low_att) 逐位置和为 1（互斥分配）。
        attention_vectors = torch.cat([high_att, low_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)  # 重新拆开

        # 按注意力权重加权各自支路
        fea_high = high * high_att
        fea_low = low * low_att

        out = self.out(fea_high + fea_low)   # 融合后过 1x1 输出
        return out

class Patch_ap(nn.Module):
    """Patch-level 的低频/高频分离（局部自适应池化）。

    与 Gap(全局) 相对，这里在空间上按 patch_size x patch_size 划分格子，
    对每个 patch 求均值作为局部"低频"，再计算局部高频残差，并分别用可学习
    系数 l(低频) 和 h(高频) 调制。用于捕捉局部纹理细节。
    """
    def __init__(self, inchannel, patch_size):
        super(Patch_ap, self).__init__()

        # 每个小 patch 上做全局平均池化。
        self.ap = nn.AdaptiveAvgPool2d((1,1))

        self.patch_size = patch_size
        # 把 patch 摊平后的总元素数（每通道的元素数）。
        self.channel = inchannel * patch_size**2
        # 每个元素一个可学习系数，初始为 0。
        self.h = nn.Parameter(torch.zeros(self.channel))  # 高频系数
        self.l = nn.Parameter(torch.zeros(self.channel))  # 低频系数

    def forward(self, x):

        # 用 einops 把空间 WxH 重新排布为 patch 的格子结构：
        #   'b c (p1 w1) (p2 w2)' -> 先把空间维度拆分出 p (patch 位置) 和 w (patch 内坐标)。
        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        # 再把每个 patch 摊平成通道维，这样每个"空间位置(w1,w2)"就是输入尺寸缩小的特征图，
        # 后续 AdaptiveAvgPool2d(1,1) 相当于对每个 patch 求均值。
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        low = self.ap(patch_x)  # 每个 patch 的局部低频（均值）
        # 高频 = 原 patch - 局部低频，乘系数 h
        high = (patch_x - low) * self.h[None, :, None, None]
        # 重建：高频 + 低频*l
        out = high + low * self.l[None, :, None, None]
        # 还原回原来的空间排布（逆变换）。
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)

        return out