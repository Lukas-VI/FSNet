"""Desnowing 主入口：构建模型，按 mode 训练或测试。

与 ITS main.py 差异：
  1) 模型保存路径为 results/FSNet/CSD/。
  2) 训练超参不同（num_epoch=2000，save/valid_freq=50）。
  3) 无 numpy/random 导入（当前未实际使用）。
"""
import os
import torch
import argparse
from torch.backends import cudnn
from models.FSNet import build_net
from train import _train
from eval import _eval
import numpy as np
import random

def main(args):
    # CUDNN
    cudnn.benchmark = True  # 固定尺寸下加速卷积

    # 创建运行所需目录
    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()
    print(model)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='FSNet', type=str)
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    # Train —— Desnowing 训练超参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=8e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=2000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--valid_freq', type=int, default=50)
    parser.add_argument('--resume', type=str, default='')

    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', 'FSNet', 'CSD/')  # CSD 雪数据集路径
    args.result_dir = os.path.join('results/', args.model_name, 'test')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    # 复制关键源码到模型保存目录，便于追溯实验代码版本
    command = 'cp ' + 'models/layers.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'models/FSNet.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'train.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)
    print(args)
    main(args)