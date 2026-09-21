# 中文导读（FSNet，图像复原学习入门）

> 以下中文导读为学习用资料，基于对代码（含已标注"理解存疑"处）与论文简述的整理。
> 原 README 的全部内容（Abstract、Installation、Training and Evaluation、Results、Citation、Contact）在本导读之后完整保留，请务必一并阅读。

## 一句话定位
FSNet（Frequency Selection Network，TPAMI 2023）是一个用于**图像复原**的通用卷积网络。核心思想是把特征分解到**不同的频段**并**按内容自适应地挑选有用频率成分**来恢复图像。本仓库覆盖**多个子任务**目录：图像去雾（Dehazing/ITS、OTS）、图像去雪（Desnowing）、图像运动去模糊（Motion_deblurring）；论文还涵盖去雨/去噪与离焦去模糊等（本仓库未全部收录）。

## 方法与核心思想
- **核心特色：动态频率选择**。与用固定小波等工具做频率分解的方法不同，FSNet 用**动态滤波器（dynamic_filter）+ 高低频分离（Gap / Patch_ap）**，让滤波核/频率贡献由**输入内容决定**：
  - `dynamic_filter`：用全局池化+1×1 卷积从输入"预测"一组滤波核，再对特征做自适应加权（k3=3×3、k5=5×5 多尺度感受野），配合 softmax 归一化。
  - `Gap`（全局高低频分离）：全局平均池化得到低频（每个通道一个常数），残差为高频，用可学习系数 `fscale_d / fscale_h` 分别缩放低/高频贡献（初始为 0）。
  - `Patch_ap`（局部 patch 版本高低频分离）：在局部 patch 上做同样的高低频拆分，捕捉局部纹理。
- **残差块结构（ResBlock，核心重复单元）**：`conv1 →(可选)动态滤波→ 通道对半拆，一半走 Gap、一半走 Patch_ap → concat → conv2 → +x`。EBlock/DBlock 中最后一个 ResBlock 才开启动态滤波（`filter=True`）。
- **最内层轻量 Unet**：EBlock1/DBlock1 在最低分辨率层再嵌一个小 U 型（内部在约 1/4 处下降采样、3/4 处上采样回原尺寸并 1×1 融合 skip），扩大感受野。
- **多尺度 U 型主体**：基础通道 32，逐级翻倍；输出 `outputs[0/1/2]` 分别对应 **1/4、1/2、原尺寸**三张复原图（残差式与输入相加）。
- **SCM + FAM 引导融合**：SCM 从低分辨率输入图提取条件特征（类 NAFNet），FAM 用通道拼接+卷积融合进主路。
- **损失**（见各 task 的 train.py）：**总损失 = 多尺度空间 L1（content）+ 0.1 × 多尺度频域 FFT L1（fft）**，空域+频域双监督。
- **优化**：Adam + 前 3 epoch 线性 warmup 后接余弦退火。

## 目录结构导读（建议按依赖关系阅读）
```
FSNet/
├── README.md                     # 本文件（中文导读 + 原英文 README）
├── pytorch-gradual-warmup-lr/    # 第三方 warmup 学习率调度器（需先 setup 安装）
├── Dehazing/
│   ├── ITS/                      # 去雾——合成室内（SOTS-Indoor），batch=32 lr=8e-4 num_epoch=1000
│   └── OTS/                      # 去雾——合成室外（SOTS-Outdoor），batch=8 lr=1e-4 num_epoch=30
│       └── 每个都是同一套文件：main.py 入口、models/FSNet.py(主网络)、models/layers.py(基础积木)、
│           train.py、valid.py、eval.py、data/(data_load.py 读图/增强)、utils.py
├── Desnowing/                    # 去雪（CSD/SRRS/Snow100K），batch=32 num_epoch=2000，含 README.md 数据说明
└── Motion_deblurring/            # 运动去模糊（GoPro/HIDE），batch=4 lr=1e-4 num_epoch=3000；含 README.md
```
**阅读顺序建议**：(1) `models/layers.py`（基础积木：dynamic_filter / Gap / Patch_ap / ResBlock / Unet）→
(2) `models/FSNet.py`（主网络 forward）→ (3) `train.py`（损失）→ (4) `main.py`（入口与超参）→ (5) `valid.py`/`eval.py`。

## 训练 / 测试如何跑
各任务目录结构一致，仅超参与保存目录不同。先安装依赖并 `pip install` / 装好 warmup 调度器（见原 README Installation）。以**去雾 ITS** 为例：
```
# 训练
python main.py --mode train --data_dir your_path/ITS --batch_size 32 --num_epoch 1000
# 测试（--test_model 指定权重；--save_image True 可存图）
python main.py --data_dir your_path/ITS --test_model path/to/model.pkl
```
- 去雪 Desnowing：`--mode train --num_epoch 2000`；运动去模糊 Motion：`--num_epoch 3000`（RSBlur 可设 710，见注释）。
- 各任务数据组织见对应 `README.md`（如 Motion 的 `your_path/GOPRO` 结构）。
- 训练产物输出到 `results/FSNet/<ITS|ots|CSD|GoPro>/`（Best.pkl / Final.pkl / model_<epoch>.pkl）。

## 学习建议 / 易踩坑（源自注释阶段的观察与 bug 线索，如实记录，供验证）
- **多尺度输出下标**：`model(x)[0/1/2]` 分别对应当前任务的 1/4、1/2、原尺寸输出；**测试/推理只用 `model(x)[2]`（原尺寸）**（见 eval.py）。
- **注意大小写/命名差异**：类名如 `Unet`、`dynamic_filter`、`Gap`、`Patch_ap` 均沿用原作者命名（大小写不规范为原作者风格），理解即可、**勿改**。
- **`Gap` 初始系数为 0 的含义**：`fscale_d` 初始为 0 表示"减掉全部低频、只留高频残差"；`fscale_h+1` 使初始等价恒等。理解高低频调制时注意这个"初始=恒等/特例"的设计。
- **动态滤波仅部分块启用**：只有每个 EBlock/DBlock（及 Unet 内）的**最后一个** ResBlock 才 `filter=True` 走动态滤波，其余为普通残差块。
- **环境版本偏老**：README 标注 PyTorch 1.8.1 / CUDA 10.2，且明确要求用 **Conda 的 pillow**（别用 pip 版）；较新环境需自行适配。
- **评估有两套 PSNR**（eval.py）：torch 的 `mse_loss` 与 skimage 的 `peak_signal_noise_ratio` 并存，最终平均统计用 torch 版；SSIM 会下采样到约 256 再算，降低算力。
- **梯度裁剪阈值小**（train.py 中 `clip_grad_norm_` 阈值 `0.001`）：若非原作者有意为之，若训练异常可验证其对训练的影响。
- **代码跨任务复制**：各任务目录是该模型同一结构的副本，逻辑一致，仅空行/命名/默认超参不同；看一份即可，易踩"改了这份忘改那份"的坑。

# Image Restoration Via Frequency Selection

Yuning Cui, Wenqi Ren, Xiaochun Cao, and Alois Knoll

>Image restoration aims to reconstruct the latent sharp image from its corrupted counterpart. Besides dealing with this longstanding task in the spatial domain, a few approaches seek solutions in the frequency domain by considering the large discrepancy between spectra of sharp/degraded image pairs. However, these algorithms commonly utilize transformation tools, e.g., wavelet transform, to split features into several frequency parts, which is not flexible enough to select the most informative frequency component to recover. In this paper, we exploit a multi-branch and content-aware module to decompose features into separate frequency subbands dynamically and locally, and then accentuate the useful ones via channel-wise attention weights. In addition, to handle large-scale degradation blurs, we propose an extremely simple decoupling and modulation module to enlarge the receptive field via global and window-based average pooling. Furthermore, we merge the paradigm of multi-stage networks into a single U-shaped network to pursue multi-scale receptive fields and improve efficiency. Finally, integrating the above designs into a convolutional backbone, the proposed Frequency Selection Network (FSNet) performs favorably against state-of-the-art algorithms on 20 different benchmark datasets for 6 representative image restoration tasks, including single-image defocus deblurring, image dehazing, image motion deblurring, image desnowing, image deraining, and image denoising.

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
conda install pillow
~~~
**Please use the *pillow* package downloaded by Conda instead of pip.**


Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Training and Evaluation
Please refer to respective directories.
## Results [Download](https://drive.google.com/drive/folders/1bZb9L660t4jL2wAqr23iTg6eyAnCuUBt?usp=sharing)
|Task|Dataset|PSNR|SSIM|
|----|------|-----|----|
|**Motion Deblurring**|GoPro|33.29|0.963|
||HIDE|31.05|0.941|
||RSBlur|34.31|0.872|
||RealBlur-R|35.84|0.952|
|**Image Dehazing**|SOTS-Indoor|42.45|0.997|
||SOTS-Outdoor|40.40|0.997|
||Dense-Haze|17.13|0.65|
||NH-HAZE|20.55|0.81|
||NHR|26.30|0.976|
||Haze4K|34.12|0.99|
|**Image Desnowing**|CSD|38.37|0.99|
||SRRS|32.33|0.98|
||Snow100K|33.76|0.95|
|**Image Deraining**|Average|33.51|0.916|
|**Defocus Deblurring**|DPDD<sub>*single*</sub>|26.22|0.811|


## Citation
~~~
@article{cui2023image,
  title={Image Restoration Via Frequency Selection},
  author={Cui, Yuning and Ren, Wenqi and Cao, Xiaochun and Knoll, Alois},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
~~~

## Contact
Should you have any question, please contact Yuning Cui.
