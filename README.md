# RFSNN: Reversed Fusion Spiking Neural Network

中文题目：脉冲神经网络理论与算法研究  
English title: Research on Spiking Neural Network Theory and Algorithm

## 项目简介

本项目围绕 **RFSNN（Reversed Fusion Spiking Neural Network）** 展开，目标是提升脉冲神经网络在动态图像/事件视觉分类任务中的多尺度时空特征融合能力与时间依赖建模能力，同时兼顾轻量化与计算效率。

根据论文内容，RFSNN 的核心思路包括：

* **反向融合 / 反向跳跃连接**：在编码器-解码器结构中引入反向跳跃连接，使浅层运动细节与深层语义信息进行跨尺度交互；
* **双阶段注意力机制**：在瓶颈处引入 **CBAM** 增强空间与通道判别性，在解码后引入 **时间自注意力** 建模长距离时间依赖；
* **微观结构调优**：优化 **LIF 神经元** 与池化层的顺序，提升时序表征能力；
* **轻量化设计**：在精度、时间步长、模型体积和参数量之间取得较好的平衡。

## 论文中报告的结果

|Dataset|Accuracy|Timesteps|Model Size|Params|
|-|-:|-:|-:|-:|
|DVS-Gesture|95.48%|20|2.89 MB|759k|
|CIFAR-10|91.13%|2|2.89 MB|759k|
|MNIST|98.36%|4|2.89 MB|759k|

## 当前已提供

* `cbtran.py`

该脚本实现了一个基于 **SpikingJelly + PyTorch** 的 DVS-Gesture 训练流程，主要包含：

* `CBAM`
* `PositionalEncoding`
* `TemporalSelfAttention`
* `DVSGestureNet`
* 基于 `DVS128Gesture` 的数据加载
* Adam / SGD 优化器
* CosineAnnealingLR 学习率调度
* TensorBoard 日志
* checkpoint 保存
* `metrics.xlsx` 导出
* Reptile-style 元学习式内环更新

目录工程包含多组实验或结果文件夹，主题包括：

* 不同层数/不同学习率实验
* `unet`
* `transformer1+U`
* `空间时间trans`
* `空间时间通道`
* `CIFAR10`
* `MNIST`
* 神经元顺序交换/上采样交换/下采样交换等消融实验

├── models/
├── datasets/
├── scripts/
├── experiments/
├── checkpoints/
├── docs/
└── README.md
```

## 方法概览

### 1\. 输入表示

对于事件流数据，采用时间窗口累积策略，将连续时间内的事件按正负极性组织为固定长度的帧序列，再构造成 `T × C × H × W` 的时空张量输入网络。

### 2\. 网络结构

当前代码中的主干为带有编码器-解码器结构的 SNN：

* 编码端：多层卷积 + BN + LIF + 下采样
* 瓶颈端：CBAM
* 解码端：反卷积上采样 + 跳跃连接特征拼接 + 卷积/LIF
* 输出端：全局池化、全连接分类头
* 时间维：Temporal Self-Attention

### 3\. 训练机制

当前 `cbtran.py` 的训练流程包含：

* 常规监督学习
* CrossEntropy loss
* CosineAnnealingLR
* 可选 AMP
* 可选 CuPy backend
* Reptile-style 内环更新

## 依赖环境

建议环境：

* Python 3.9+
* PyTorch
* SpikingJelly
* pandas
* tensorboard
* CUDA（可选）
* CuPy（可选）

可参考安装：

```bash
pip install torch torchvision torchaudio
pip install spikingjelly pandas tensorboard
```

若需要 CuPy 后端，请根据本机 CUDA 版本安装对应的 CuPy 包。

## 数据集准备

当前脚本默认使用：

* **DVS128Gesture / DVS-Gesture**

在 `cbtran.py` 中，默认数据路径为：

```text
D:/datasets/DVS128Gesture
```

发布到 GitHub 时，建议改成命令行传参或相对路径，例如：

```bash
python cbtran.py -data-dir ./data/DVS128Gesture
```

## 训练示例

```bash
python cbtran.py \\
  -T 16 \\
  -device cuda \\
  -b 4 \\
  -epochs 400 \\
  -j 4 \\
  -data-dir ./data/DVS128Gesture \\
  -out-dir ./output/cbam\_reptile \\
  -opt adam \\
  -lr 0.0008 \\
  -channels 64 \\
  --inner-lr 1e-2 \\
  --inner-steps 1 \\
  --meta-lr 1e-3
```

## 输出文件

* `checkpoint\_max.pth`
* `checkpoint\_latest.pth`
* `metrics.xlsx`
* `args.txt`
* TensorBoard 日志文件

## 

