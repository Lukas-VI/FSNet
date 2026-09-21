import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from pytorch_msssim import ssim
import torch.nn.functional as f

from skimage import img_as_ubyte
import cv2

def _eval(model, args):
    """测试（评估）主函数：加载预训练模型，对测试集逐张推理并计算 PSNR / SSIM。

    - 推理前会给输入补 0 到 factor=8 的整数倍（模型多尺度结构对输入尺寸有整除要求），推理后再裁回原尺寸。
    - 使用第三个输出（原分辨率复原图，pred[2]）作为最终结果。
    """
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])  # 加载模型权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()  # 用于统计单张推理耗时均值
    model.eval()  # 切换评估模式（关闭 BN dropout 等）
    factor = 8  # 补零对齐因子
    with torch.no_grad():  # 不计算梯度，省显存提速
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            # ---- 尺寸对齐：把 H/W 补成 factor 的整数倍 ----
            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')  # 反射填充

            tm = time.time()

            pred = model(input_img)[2]  # 取原分辨率输出
            pred = pred[:,:,:h,:w]     # 裁掉填充部分

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)  # 裁剪到 [0,1]

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            # ---- PSNR ----
            label_img = (label_img).cuda()
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))  # 基于 MSE 的 PSNR
            # ---- SSIM（pytorch_msssim）----
            # 根据图尺寸决定下采样比例，把长边降到 ~256 再算 SSIM，避免超大图显存过大。
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))),
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False)
            print('%d iter PSNR_dehazing: %.2f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
            ssim_adder(ssim_val)

            # ---- 可选：保存复原图 ----
            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255  # 反量化+0.5 偏移，转为 8bit 时避免四舍五入偏差
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)  # 另一种 PSNR 实现
            psnr_adder(psnr_val)

            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))

        # ---- 汇总打印均值 ----
        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.5f dB' % (ssim_adder.average()))

        print("Average time: %f" % adder.average())