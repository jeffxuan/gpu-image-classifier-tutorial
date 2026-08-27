# 我的犬种图像分类学习记录

这份记录总结了从远程 GPU 环境配置，到训练、评估和部署犬种分类器的完整过程。

## 1. 项目目标

使用 Kaggle Dog Breed Identification 数据集训练一个图像分类模型。输入一张狗的照片，模型从 120 个犬种中给出最可能的类别，并显示 Top-5 预测结果。

## 2. 远程环境

- 主机：Linux GPU 主机，通过 VS Code Remote-SSH 连接
- GPU：NVIDIA GeForce RTX 4090
- Python：3.10
- 虚拟环境：`/home/xuan/venvs/gpu-classifier`
- PyTorch：`2.10.0+cu130`
- TorchVision：`0.25.0+cu130`

激活环境：

```bash
source /home/xuan/venvs/gpu-classifier/bin/activate
cd /home/xuan/dog-breed-classifier
```

检查版本：

```bash
python --version
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

## 3. 安装 PyTorch 时遇到的问题

最初直接安装 wheel 时使用了 `--no-deps`。这个选项会跳过 PyTorch 依赖的 NVIDIA CUDA 运行时包，例如 CUDA Runtime 和 cuBLAS，结果是运行时找不到类似 `libcudart.so.13` 的动态库。

解决方法是保留依赖安装，让 pip 从镜像源获取所需包：

```bash
/usr/bin/pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --target /home/xuan/venvs/gpu-classifier/lib/python3.10/site-packages \
  /home/xuan/gpu-classifier/wheels/torch-2.10.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl \
  /home/xuan/gpu-classifier/wheels/torchvision-0.25.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl
```

这里的关键经验是：PyTorch、TorchVision、Python 和 CUDA 构建版本必须相互匹配，而且不要随意跳过运行时依赖。

## 4. 数据准备

数据集包含 120 个犬种、10,222 张带标签的训练图片。按照类别分层划分：

- 训练集：8,185 张
- 验证集：2,037 张
- 类别数：120

分层划分能避免某些小样本犬种只出现在训练集或验证集，保证评估结果更可信。

对于每个类别图片数量较少的问题，采用以下策略保证质量：

- 使用分层划分，而不是完全随机划分
- 训练阶段使用数据增强，例如随机裁剪、水平翻转和颜色变化
- 验证集不使用随机增强，保证结果稳定
- 观察每个类别的准确率，而不只看总体准确率
- 使用混淆矩阵检查外观相似犬种之间的错误

## 5. 模型训练

模型使用 ImageNet 预训练的 ResNet-34，并将最后的全连接层替换为 120 类分类头。

训练分为两个阶段：

1. 先冻结主干网络，只训练分类头，验证准确率达到约 `82.18%`。
2. 再解冻 `layer4` 和分类头进行微调，验证准确率达到约 `84.68%`。

最终模型保存为：

```text
checkpoints/resnet34_finetune_best.pt
```

只保存验证集表现最好的 checkpoint，避免最后一个 epoch 的模型反而变差。

## 6. 结果分析

验证集分析中较常见的混淆包括：

- `appenzeller` 预测为 `entlebucher`
- `staffordshire_bullterrier` 预测为 `american_staffordshire_terrier`
- `lakeland_terrier` 预测为 `airedale`
- `silky_terrier` 预测为 `australian_terrier`
- `rhodesian_ridgeback` 预测为 `vizsla`

这些错误说明模型在外观相似犬种之间仍有提升空间。下一步可以尝试更强的数据增强、类别均衡采样、更高分辨率或更大的预训练模型。

## 7. 用模型预测具体图片

推理时必须使用和训练阶段一致的预处理：调整尺寸、中心裁剪、转为 Tensor，并使用 ImageNet 均值和标准差归一化。模型输出经过 softmax 后，取概率最高的 5 个类别。

推理的基本逻辑是：

```python
with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)
    scores, indices = torch.topk(probabilities, k=5)
```

## 8. Gradio 图形界面

为了方便测试，使用 Gradio 创建上传图片和显示 Top-5 结果的界面。通过 VS Code 的端口转发访问：

```text
http://127.0.0.1:7860
```

曾遇到的错误：

```text
PermissionError: [Errno 13] Permission denied: '/tmp/gradio/...'
```

原因是 Gradio 上传文件时尝试在共享临时目录创建文件夹，但当前服务器用户没有权限。解决思路是改用用户自己的临时目录：

```bash
mkdir -p "$HOME/.gradio_tmp"
GRADIO_TEMP_DIR="$HOME/.gradio_tmp" \
TMPDIR="$HOME/.gradio_tmp" \
python app.py
```

这个问题和模型本身、GPU 或 CUDA 无关，而是 Web 界面的临时文件权限问题。

## 9. 当前成果

- 完成 120 类犬种数据准备
- 完成 ResNet-34 GPU 训练
- 最佳验证准确率约 `84.68%`
- 生成验证集预测和混淆分析结果
- 生成 Kaggle 提交文件
- 完成 Gradio 图形化预测界面
- 将代码和学习记录上传到 GitHub

## 10. 重要经验

1. 先确认 Python、PyTorch、TorchVision 和 CUDA 的版本关系。
2. 遇到安装错误时，先看完整 traceback，不要只根据浏览器上的 500 错误猜原因。
3. 训练准确率不能代表全部质量，要同时查看验证集、类别准确率和混淆矩阵。
4. 小样本类别要通过分层划分、数据增强和逐类评估来控制风险。
5. 代码、环境说明和实验记录应该一起保存，方便复现和向导师汇报。
