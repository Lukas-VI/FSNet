import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f


def _valid(model, args, ep):
    """训练过程中周期性的验证：在验证集上计算平均 PSNR（供 _train 决定是否保存 Best.pkl）。

    与 eval.py 的区别：只算 PSNR（不算 SSIM），且结束后把模型切回 train 模式继续训练。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    its = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)  # 验证集可复用测试目录
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 8
        for idx, data in enumerate(its):
            input_img, label_img = data
            input_img = input_img.to(device)

            # 尺寸补齐到 factor 的整数倍（与 eval 一致）
            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            # 每个验证 epoch 建一个结果子目录（实际此处只算指标未保存图）
            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)[2]  # 取原分辨率输出
            pred = pred[:,:,:h,:w]     # 裁掉填充

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)  # skimage 版 PSNR

            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')  # 打印进度（不换行）

    print('\n')
    model.train()  # 验证完切回训练模式
    return psnr_adder.average()