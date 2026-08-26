# GPU Image Classifier Tutorial

一份从零开始使用 Linux GPU 训练图像分类器的实践记录。项目使用 PyTorch、torchvision 和 Fashion-MNIST，在 NVIDIA GPU 上训练一个简单的 CNN。

## 最终环境

```text
OS: Linux x86_64
Python: 3.10.12
GPU: NVIDIA GeForce RTX 4090
PyTorch: 2.10.0+cu130
torchvision: 0.25.0+cu130
CUDA runtime: 13.0
```

## 1. 通过 VS Code 连接 GPU 主机

在 VS Code 安装 Remote - SSH 扩展，然后连接：

```text
xuan@YOUR_GPU_HOST
```

不要把真实主机地址、密码或 SSH 私钥提交到 GitHub。

连接后在远程终端确认主机和 GPU：

```bash
whoami
hostname
nvidia-smi
```

## 2. 创建并激活虚拟环境

```bash
python3 --version
python3 -m venv /home/xuan/venvs/gpu-classifier --without-pip
source /home/xuan/venvs/gpu-classifier/bin/activate
```

激活后检查：

```bash
which python
python --version
```

预期路径：

```text
/home/xuan/venvs/gpu-classifier/bin/python
Python 3.10.12
```

每次重新打开终端都需要重新激活：

```bash
source /home/xuan/venvs/gpu-classifier/bin/activate
```

退出环境：

```bash
deactivate
```

## 3. PyTorch 版本匹配

本项目使用以下匹配组合：

```text
torch       2.10.0+cu130
torchvision 0.25.0+cu130
Python      cp310
平台        manylinux_2_28_x86_64
```

`cu130` 表示 PyTorch 包内的 CUDA 13 构建版本。主机驱动需要支持运行 CUDA 13；使用 `nvidia-smi` 检查驱动状态。

## 4. 安装时遇到的问题

### 虚拟环境没有 pip

由于系统没有 `python3-venv` 或 `ensurepip`，环境最初用 `--without-pip` 创建。因此：

```bash
python -m pip
```

可能报错：

```text
No module named pip
```

临时解决方式是使用系统 pip，并通过 `--target` 把包安装到虚拟环境：

```bash
/usr/bin/pip3 install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --target /home/xuan/venvs/gpu-classifier/lib/python3.10/site-packages \
  /home/xuan/gpu-classifier/wheels/torch-2.10.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl \
  /home/xuan/gpu-classifier/wheels/torchvision-0.25.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl
```

### 不要使用 `--no-deps`

只安装 torch 和 torchvision 而跳过依赖，会导致：

```text
libcudart.so.13: cannot open shared object file
libcublas.so.* not found
```

原因是 CUDA runtime、cuBLAS、cuDNN、NCCL、Triton 等依赖没有安装。安装 PyTorch wheel 时应允许 pip 安装依赖：

```bash
/usr/bin/pip3 install --upgrade \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --target /home/xuan/venvs/gpu-classifier/lib/python3.10/site-packages \
  /home/xuan/gpu-classifier/wheels/torch-2.10.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl \
  /home/xuan/gpu-classifier/wheels/torchvision-0.25.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl
```

清华镜像成功安装了所需的 CUDA 13 依赖。本机无法联网时，也可以在有网络的电脑下载 Linux x86_64 wheels，再用 `scp` 上传。

### wheel 文件名必须规范

文件名中的 `(2)` 会导致 pip 认为它不是合法 wheel。应改成标准名称：

```bash
mv 'torch-2.10.0+cu130-cp310-cp310-manylinux_2_28_x86_64 (2).whl' \
   'torch-2.10.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl'
```

## 5. 验证 PyTorch 和 GPU

```bash
source /home/xuan/venvs/gpu-classifier/bin/activate

python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期结果包含：

```text
2.10.0+cu130
0.25.0+cu130
13.0
True
NVIDIA GeForce RTX 4090
```

## 6. Fashion-MNIST CNN 分类器

Fashion-MNIST 包含 60,000 张训练图片和 10,000 张测试图片。每张图片是 28x28 灰度图，共 10 个服装类别。

项目目录建议如下：

```text
gpu-classifier/
├── data/
├── checkpoints/
│   └── fashion_mnist_cnn.pt
├── train.py
└── predict.py
```

训练脚本需要完成：

1. 下载并预处理 Fashion-MNIST。
2. 使用卷积层提取图片特征。
3. 把模型、图片和标签放到 CUDA GPU。
4. 使用交叉熵损失和 Adam 优化器训练。
5. 在测试集上计算准确率。
6. 保存模型参数。

运行：

```bash
source /home/xuan/venvs/gpu-classifier/bin/activate
cd /home/xuan/gpu-classifier
python train.py
```

一次实际训练得到的结果是：

```text
Epoch 1/5 | Test accuracy: 87.85%
Epoch 2/5 | Test accuracy: 88.59%
Epoch 3/5 | Test accuracy: 89.75%
Epoch 4/5 | Test accuracy: 90.87%
Epoch 5/5 | Test accuracy: 91.08%
```

模型保存到：

```text
checkpoints/fashion_mnist_cnn.pt
```

## 7. 查看具体图片的预测

`predict.py` 加载保存的模型，从测试集选取一个编号，并打印真实类别、预测类别和置信度：

```bash
python predict.py
```

修改脚本中的：

```python
index = 100
```

即可查看另一张测试图片。测试集编号范围是 `0` 到 `9999`。

## 8. 常用诊断命令

```bash
which python
python --version
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
du -sh data checkpoints
ls -lh checkpoints
```

## 9. 安全注意事项

- 不要提交密码、SSH 私钥、token 或服务器真实凭据。
- 不要把带有个人路径和内网地址的截图提交到公开仓库。
- 公开教程中使用 `YOUR_GPU_HOST`、`YOUR_USERNAME` 等占位符。
- wheel 文件和数据集通常很大，不建议直接提交到 Git 仓库。

## 10. 后续实验

- 增加训练轮数并比较准确率。
- 调整 batch size 和学习率。
- 加入数据增强。
- 绘制 loss 和 accuracy 曲线。
- 比较 CPU 与 GPU 训练时间。
- 分析哪些服装类别最容易混淆。
