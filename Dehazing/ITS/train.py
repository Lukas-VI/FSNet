import os
import torch
from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn

from warmup_scheduler import GradualWarmupScheduler

def _train(model, args):
    """FSNet 训练主函数（单卡）。

    训练要点：
      1) 损失 = 空间 L1（三个尺度） + 0.1 * 频域 FFT L1（三个尺度），既管像素又管频率。
      2) 学习率先 warmup 3 个 epoch，再走余弦退火(CosineAnnealing)。
      3) 每轮反向传播前对梯度做 clip（防止梯度爆炸）。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()  # 核心损失，空间域和频域共用

    # Adam 优化器，betas 默认参数。注意 args.weight_decay 虽定义了但此处未使用。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)  # 一个 epoch 内的迭代次数 = 样本数/ batch_size
    warmup_epochs=3  # 学习率预热(warm up)的 epoch 数
    # 后半段用余弦退火：从初始 lr 平滑降到 eta_min，总的退火区间是 num_epoch-warmup。
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
    # 先 warmup 后接 cosine：multiplier=1 表示 warmup 结束时把 lr 提到初始 lr。
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()  # 手动走一步，让 warmup 生效
    epoch = 1
    if args.resume:
        # 断点续训：加载之前保存的模型/优化器/epoch。
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1  # 从下一个 epoch 继续

    writer = SummaryWriter()  # tensorboard 记录
    # 分别累计 epoch / iter 级别的像素损失与 FFT 损失的滑动均值（Adder 见 utils）。
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')  # 计时（分钟）
    iter_timer = Timer('m')
    best_psnr=-1  # 记录历史最佳 PSNR，用于保存 Best.pkl

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)  # 得到三个分辨率的预测
            # 把 GT 缩小到对应分辨率，与各级预测对齐（0/1/2 对应 1/4/1/2/原尺寸）。
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)  # 1/4 分辨率
            l2 = criterion(pred_img[1], label_img2)  # 1/2 分辨率
            l3 = criterion(pred_img[2], label_img)   # 原分辨率
            loss_content = l1+l2+l3  # 空间域 L1 损失（多尺度累加）

            # ---- 频域（FFT）损失：一致地对预测与 GT 做傅里叶变换后算 L1 ----
            # 把复数拆成实部/虚部拼成最后一维，方便用 L1Loss 比较。
            label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2,-1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2,-1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2,-1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3  # 频域多尺度损失

            # 总损失 = 空间损失 + 0.1 * 频域损失
            loss = loss_content + 0.1 * loss_fft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)  # 梯度裁剪
            optimizer.step()

            # 累计 iter 级与 epoch 级损失均值，用于打印
            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0:
                # 每个 print_freq 迭代打印一次进度与平均损失，并写入 tensorboard
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0], iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)

                iter_timer.tic()  # 打印后重置，统计下一个打印间隔耗时
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
        # ---- 每个 epoch 结束后：保存模型 ----
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)  # 每个完整 epoch 都覆盖保存（含优化器，可续训）

        if epoch_idx % args.save_freq == 0:
            # 每隔 save_freq 个 epoch 额外保存一份带 epoch 编号的快照
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()  # 学习率调度走一步

        # ---- 周期性验证 ----
        if epoch_idx % args.valid_freq == 0:
            val = _valid(model, args, epoch_idx)  # 在验证集上算平均 PSNR
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:  # 刷新最佳 PSNR 则保存 Best.pkl
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    # 训练全部结束后保存最终模型
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)