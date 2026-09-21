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
    """入口函数：构建模型，根据 args.mode 选择训练或测试。"""
    # CUDNN
    cudnn.benchmark = True  # 输入尺寸固定时加速卷积

    # 创建运行所需目录（结果/模型保存/输出）
    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()  # 构建 FSNet
    print(model)

    if torch.cuda.is_available():
        model.cuda()  # 有 GPU 则搬到 GPU（注意：仅单卡）
    if args.mode == 'train':
        _train(model, args)  # 训练

    elif args.mode == 'test':
        _eval(model, args)   # 测试


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories  —— 相关目录/名称参数
    parser.add_argument('--model_name', default='FSNet', type=str)

    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='')

    # Train —— 训练超参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=8e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--valid_freq', type=int, default=20)
    parser.add_argument('--resume', type=str, default='')


    # Test —— 测试相关参数
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    # 固定模型/结果保存路径（这里硬编码了 ITS 任务名）
    args.model_save_dir = os.path.join('results/', 'FSNet', 'ITS/')
    args.result_dir = os.path.join('results/', args.model_name, 'test')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    # 把关键源码复制到模型保存目录，方便日后追溯该次实验所用的代码版本
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