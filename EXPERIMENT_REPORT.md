# Fashion-MNIST GPU 图像分类实验报告

## 摘要

本实验在 Linux GPU 主机上使用 PyTorch 构建并训练一个卷积神经网络（CNN），完成 Fashion-MNIST 十分类任务。实验主机配备 NVIDIA GeForce RTX 4090，模型使用 CUDA 加速训练。

## 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Linux x86_64 |
| Python | 3.10.12 |
| GPU | NVIDIA GeForce RTX 4090 |
| PyTorch | 2.10.0+cu130 |
| torchvision | 0.25.0+cu130 |
| CUDA runtime | 13.0 |
| 数据集 | Fashion-MNIST |

## 任务说明

Fashion-MNIST 包含 60,000 张训练图片和 10,000 张测试图片。每张图片为 28x28 灰度图，目标是将图片分类为以下十个服装类别之一：

```text
T-shirt/top, Trouser, Pullover, Dress, Coat,
Sandal, Shirt, Sneaker, Bag, Ankle boot
```

## 模型和训练方法

模型由两组卷积、ReLU 激活和最大池化层组成，随后连接两个全连接层，最后输出十个类别分数。训练使用：

```text
损失函数：CrossEntropyLoss
优化器：Adam
学习率：0.001
训练轮数：5
训练 batch size：128
```

图片经过 `ToTensor` 和归一化处理。模型、输入图片和标签被放到 CUDA GPU 上进行计算。

## Baseline 结果

以下结果来自第一次成功的 GPU 训练：

| Epoch | Loss | Train accuracy | Test accuracy |
|---:|---:|---:|---:|
| 1 | 0.4841 | 82.63% | 87.85% |
| 2 | 0.3050 | 89.02% | 88.59% |
| 3 | 0.2571 | 90.59% | 89.75% |
| 4 | 0.2257 | 91.74% | 90.87% |
| 5 | 0.2024 | 92.55% | 91.08% |

最终测试准确率为 **91.08%**。训练过程中 loss 持续下降，测试准确率持续上升，说明模型有效学习了服装图像特征。训练集准确率略高于测试集准确率，但差距不大，当前没有明显的严重过拟合现象。

## 结果分析

模型已经能够较好地完成基本服装分类任务。卷积层可以提取边缘、轮廓和纹理等局部特征，再由全连接层完成类别判断。RTX 4090 能够执行 CUDA 计算，证明 Python、PyTorch、CUDA runtime 和显卡驱动已经正确连接。

当前实验仍有三个限制：

1. 第一次训练没有保存每个 batch 或 epoch 的日志，因此暂时只有终端结果，尚未绘制曲线。
2. 尚未统计每个类别的准确率和混淆矩阵，因此还不知道哪些类别最容易混淆。
3. 尚未进行 CPU/GPU 用时对比，也没有比较不同网络结构。

## 后续实验计划

下一轮实验应在训练脚本中保存 `history.csv`，并在测试阶段保存所有预测结果。之后生成：

```text
loss_accuracy.png       训练曲线
confusion_matrix.png    混淆矩阵
class_accuracy.csv      每个类别的准确率
error_samples/          错误预测图片
```

建议进一步比较以下配置：

```text
baseline CNN
baseline + Dropout
baseline + BatchNorm
不同 batch size
CPU 与 GPU 训练时间
```

## 结论

本项目已经完成从远程 GPU 环境配置、PyTorch 安装、数据加载、CNN 建模、CUDA 训练到模型保存的完整流程。Baseline 模型在 Fashion-MNIST 测试集上达到 91.08% 准确率，说明该环境和训练流程可以用于后续更复杂的图像分类实验。

## 复现实验

```bash
source /home/xuan/venvs/gpu-classifier/bin/activate
cd /home/xuan/gpu-classifier
python train.py
python predict.py
```

运行时请根据实际目录修改路径，不要把服务器密码、SSH 私钥或内网地址提交到公开仓库。
