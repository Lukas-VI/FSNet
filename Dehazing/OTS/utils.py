import time
import numpy as np


class Adder(object):
    """累加器：统计累加值与平均值的工具类，常用于记录损失/指标的滑动均值。
    调用方式：adder(值) 累加一次；adder.average() 取当前平均值；adder.reset() 清零。
    """
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1   # 累加次数
        self.num += num   # 累加数值

    def average(self):
        return self.num / self.count


class Timer(object):
    """计时器工具，用于统计一段操作耗时。
    option 决定计时的单位：'s'秒 / 'm'分钟 / 其他为小时。
    """
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()  # 记录开始时间

    def toc(self):
        return (time.time() - self.tm) / self.devider  # 返回经过的时长（单位换算）


def check_lr(optimizer):
    """取优化器当前学习率（返回最后一个 param_group 的 lr）。"""
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr