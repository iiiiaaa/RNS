import os
import glob
import random
import shutil

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model


# ==========================================
# 环境配置 (Configuration)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16


# 固定种子保证实验是可复现
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(0)

# ==========================================
# 统计参数
# ==========================================
cifar_10_mean = (0.491, 0.482, 0.447)
cifar_10_std = (0.202, 0.199, 0.201)

mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

# 攻击参数


epsilon = 8 / 255 / 0.2
alpha = 0.8 / 255 / 0.2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])


# ==========================================
# 数据集定义
# ==========================================
class AdvDataset(Dataset):

    # 初始化（扫描数据目录 -> 收集所有图片路径 -> 生成标签 -> 保存，供后面按需读取）
    def __init__(self, data_dir, transform):
        self.images = []  # 图片路径
        self.labels = []
        self.names = []
        # 防止目录不存在报错
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} not found. Create dummy data for testing logic.")
            return

        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        self.transform = transform

    # 处理数据
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label

    def __getname__(self):
        return self.names

    def __len__(self):
        return len(self.images)


# ==========================================
# 集成模型 (ensembleNet 类)
# ==========================================
class ensembleNet(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        # 加载所有指定的模型
        self.models = nn.ModuleList([ptcv_get_model(name, pretrained=True) for name in model_names])

    def forward(self, x):
        # 初始化 logits (预测分数)
        ensemble_logits = 0

        # 遍历每一个模型
        for i, m in enumerate(self.models):
            # 累加每个模型的预测结果
            ensemble_logits += m(x)

        # 取平均值 (除以模型数量)
        # return ensemble_logits  # 其实直接返回总和也可以，梯度方向是一样的
        return ensemble_logits / len(self.models)


# ==========================================
# 攻击算法
# ==========================================

# --- 噪声添加函数 ---
def add_random_noise(x, noise_scale=0.1):
    """
    给输入张量添加高斯噪声
    Args:
        x: 输入图像 Tensor (Batch, C, H, W)
        noise_scale: 噪声的强度系数
    """
    noise = torch.randn_like(x) * noise_scale
    return x + noise


# --- 1. FGSM (无噪声) ---
def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone()
    x_adv.requires_grad = True

    loss = loss_fn(model(x_adv), y)
    loss.backward()  # 反向传播
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv


def fgsm_noisy(model, x, y, loss_fn, epsilon=epsilon, noise_scale=0.1):
    x_adv = x.detach().clone()
    x_adv.requires_grad = True

    # 【关键修改】：在计算梯度前，给输入加噪声
    # 注意：我们希望基于有噪声的输入计算梯度，从而跳出局部极值点
    x_noisy = add_random_noise(x_adv, noise_scale)

    loss = loss_fn(model(x_noisy), y)  # 输入变成 noisy
    loss.backward()

    grad = x_adv.grad.detach()  # 获取对原图 x_adv 的梯度
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv


# --- 2. I-FGSM (无噪声) ---
def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10):
    x_adv = x.detach().clone()
    for i in range(num_iter):
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = x + delta
        x_adv = x_adv.detach()
    return x_adv


def ifgsm_noisy(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, noise_scale=0.1):
    x_adv = x.detach().clone()
    for i in range(num_iter):
        x_adv.requires_grad = True

        # 【关键修改】：每次迭代前加噪声
        x_noisy = add_random_noise(x_adv, noise_scale)

        # 计算 Loss 时使用的是加噪后的图，这有助于平滑梯度
        outputs = model(x_noisy)
        loss = loss_fn(outputs, y)
        loss.backward()

        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = x + delta
        x_adv = x_adv.detach()
    return x_adv


# --- 3. MI-FGSM (无噪声 & 有噪声) ---
def mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x.detach().clone()
    momentum = torch.zeros_like(x).detach().to(device)
    for i in range(num_iter):
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        loss.backward()
        grad = x_adv.grad.detach()

        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=1, dim=1).view(-1, 1, 1, 1)
        grad_normalized = grad / (grad_norm + 1e-10)
        momentum = decay * momentum + grad_normalized

        x_adv = x_adv + alpha * momentum.sign()
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = x + delta
        x_adv = x_adv.detach()
    return x_adv


def mifgsm_noisy(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=1.5, N=5):
    if N <= 0:
        return mifgsm(
            model,
            x,
            y,
            loss_fn,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            decay=decay
        )

    x_adv = x.detach().clone()
    momentum = torch.zeros_like(x).detach().to(device)
    for i in range(num_iter):
        x_adv.requires_grad = True

        gg = 0
        for j in range(N):
            x_noisy = x_adv.detach() + torch.randn_like(x).uniform_(-epsilon * beta, epsilon * beta)
            x_noisy.requires_grad = True
            outputs = model(x_noisy)
            loss = loss_fn(outputs, y)
            loss.backward()
            grad1 = x_noisy.grad.detach()
            gg += grad1
        grad = gg/N

        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=1, dim=1).view(-1, 1, 1, 1)
        grad_normalized = grad / (grad_norm + 1e-10)
        momentum = decay * momentum + grad_normalized

        x_adv = x_adv + alpha * momentum.sign()
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = x + delta
        x_adv = x_adv.detach()

    return x_adv


# --- 4. PGD (Projected Gradient Descent, 2018) ---
# 特点：在 I-FGSM 的基础上加入了随机初始化 (Random Start)。
def pgd(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10):
    x_adv = x.detach().clone()

    # --- 生成 [-epsilon, epsilon] 范围的随机噪声 ---
    # torch.rand_like 生成[0, 1) 的均匀分布
    # (rand * 2 - 1) 变成[-1, 1) 的均匀分布
    # 再乘以 epsilon，完美支持 epsilon 是多维 Tensor 的情况
    random_noise = (torch.rand_like(x_adv) * 2 - 1) * epsilon
    x_adv = x_adv + random_noise

    delta = x_adv - x
    delta = torch.max(torch.min(delta, epsilon), -epsilon)
    x_adv = x + delta
    x_adv = x_adv.detach()

    for i in range(num_iter):
        x_adv.requires_grad = True

        # 前向传播与计算损失
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()

        # 根据梯度方向更新对抗样本
        x_adv = x_adv + alpha * grad.sign()

        # 限制扰动范围在 [-epsilon, epsilon] 之间
        delta = x_adv - x
        delta = torch.max(torch.min(delta, epsilon), -epsilon)
        x_adv = x + delta

        x_adv = x_adv.detach()

    return x_adv


# --- 5. NI-FGSM (Nesterov Accelerated Gradient FGSM, 2020) ---
def nifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x.detach().clone()
    momentum = torch.zeros_like(x).detach().to(device)

    for i in range(num_iter):
        # 核心修改：利用上一步的 momentum “提前走一步” (Look-ahead)
        x_nes = x_adv + alpha * decay * momentum
        x_nes = x_nes.detach()
        x_nes.requires_grad = True

        outputs = model(x_nes)
        loss = loss_fn(outputs, y)
        loss.backward()
        grad = x_nes.grad.detach()

        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=1, dim=1).view(-1, 1, 1, 1)
        grad_normalized = grad / (grad_norm + 1e-10)
        momentum = decay * momentum + grad_normalized

        # 更新原本的 x_adv
        x_adv = x_adv + alpha * momentum.sign()
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = x + delta
        x_adv = x_adv.detach()
    return x_adv


# --- 6. GRA (Gradient Relevance Attack, ICCV 2023) ---
def gra(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, eta=0.9):
    x_adv = x.detach().clone()
    momentum = torch.zeros_like(x).detach().to(device)

    # GRA 专属变量：记录前一次梯度的符号，以及每个像素的独立步长
    prev_grad_sign = torch.zeros_like(x).detach().to(device)
    step_size = torch.ones_like(x).detach().to(device) * alpha

    for i in range(num_iter):
        x_adv.requires_grad = True

        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        loss.backward()
        grad = x_adv.grad.detach()

        # --- 符号波动衰减逻辑 ---
        curr_grad_sign = grad.sign()
        # 判断：如果当前梯度符号与上一次不一致（且上一次非0），则说明发生震荡
        fluctuation = (curr_grad_sign != prev_grad_sign) & (prev_grad_sign != 0)
        # 对发生震荡的像素点步长乘以衰减系数 eta
        step_size[fluctuation] *= eta
        prev_grad_sign = curr_grad_sign

        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=1, dim=1).view(-1, 1, 1, 1)
        grad_normalized = grad / (grad_norm + 1e-10)
        momentum = decay * momentum + grad_normalized

        # 使用动态衰减的步长 (step_size) 进行更新，而不是固定的 alpha
        x_adv = x_adv + step_size * momentum.sign()
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = x + delta
        x_adv = x_adv.detach()
    return x_adv


# 测试模型表现
def epoch_benign(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)


# 保存到硬盘
def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    # 如果目标文件夹不存在，先把原文件夹的结构整个复制过去
    if os.path.exists(adv_dir) is not True:
        _ = shutil.copytree(data_dir, adv_dir)

    # 遍历每一张生成的攻击图片，覆盖保存
    for example, name in zip(adv_examples, adv_names):
        im = Image.fromarray(example.astype(np.uint8))  # 转成整数格式
        im.save(os.path.join(adv_dir, name))  # 保存


# 生成对抗样本
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    adv_examples = None  # 用于存放最终的所有对抗样本数组

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # 1. 执行攻击生成对抗样本 (Tensor格式)
        x_adv = attack(model, x, y, loss_fn)

        # 2. === 后处理：把 Tensor 还原成可保存的图片数组 ===

        # 反归一化 (如果数据预处理时没做 Normalize，请去掉 * std + mean)
        # 假设 mean 和 std 是全局变量
        adv_ex = ((x_adv) * std + mean).clamp(0, 1)

        # 映射到 0-255 并转为 Numpy
        adv_ex = (adv_ex * 255).clamp(0, 255)
        adv_ex = adv_ex.detach().cpu().numpy().round()

        # 调整维度: (Batch, Channel, Height, Width) -> (Batch, Height, Width, Channel)
        adv_ex = adv_ex.transpose((0, 2, 3, 1))

        # 3. 拼接当前批次的结果
        if i == 0:
            adv_examples = adv_ex
        else:
            adv_examples = np.concatenate((adv_examples, adv_ex), axis=0)

    return adv_examples
