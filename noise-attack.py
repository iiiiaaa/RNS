import os

import torch.nn as nn
from torch.utils.data import DataLoader
from noise import device, ptcv_get_model, fgsm_noisy, AdvDataset, transform, \
    fgsm, ifgsm, ifgsm_noisy, mifgsm, mifgsm_noisy, ensembleNet, epoch_benign, gen_adv_examples, create_dir, batch_size, \
    nifgsm, gra, pgd
from pytorchcv.model_provider import get_model as ptcv_get_model
from functools import partial


# 生成对抗样本
def run_attack_pipeline(model_name, attack_func, attack_name, data_root='./data'):
    """
    通用对抗样本生成流水线

    Args:
        model_name (str): 模型名称，用于加载模型和命名文件夹 (e.g., 'resnet110_cifar10')
        attack_func (function): 攻击算法函数本身 (e.g., fgsm)
        attack_name (str): 攻击算法的名称字符串，用于命名文件夹 (e.g., 'fgsm')
        data_root (str): 数据集根目录
    """

    # --- 1. 准备路径名称 ---
    save_folder_name = f"{model_name}_{attack_name}"
    output_dir = os.path.join(os.path.dirname(data_root), save_folder_name)

    # --- 2. 加载数据 ---
    print("1. 正在加载数据...")
    adv_set = AdvDataset(data_root, transform=transform)
    adv_names = adv_set.__getname__()
    adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)

    # --- 3. 加载模型 ---
    print(f"2. 正在加载模型: {model_name} ...")
    model = ptcv_get_model(model_name, pretrained=True).to(device)
    model.eval()  # 设置 eval 模式
    loss_fn = nn.CrossEntropyLoss()

    # --- 4. 生成对抗样本 ---
    print(f"3. 正在执行攻击算法: {attack_name} ...")
    # 这里调用传入的 attack_func
    adv_examples = gen_adv_examples(model, adv_loader, attack_func, loss_fn)

    # --- 5. 保存结果 ---
    create_dir(data_root, output_dir, adv_examples, adv_names)


def evaluate_transferability(adv_image_dir, target_model_list):
    """
    测试对抗样本在多个目标模型上的迁移攻击效果。

    Args:
        adv_image_dir (str): 对抗样本图片的存储目录 (例如: './data/resnet110_fgsm_adv')
        target_model_list (list): 需要测试的目标模型名称列表 (例如: ['resnet20_cifar10', 'vgg16_cifar10'])

    Returns:
        results (dict): 包含每个模型测试准确率的字典
    """
    print(f"[*] 对抗样本来源: {adv_image_dir}")
    print(f"[*] 待测模型: {len(target_model_list)}")

    # 数据加载
    adv_set = AdvDataset(adv_image_dir, transform=transform)
    adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    results = {}

    # --- 2. 遍历每一个目标模型进行测试 便利---
    for model_name in target_model_list:

        try:
            # 加载模型
            model = ptcv_get_model(model_name, pretrained=True).to(device)
            model.eval()

            acc, loss = epoch_benign(model, adv_loader, loss_fn)

            results[model_name] = acc, loss
            print(f"    [结果] 模型: {model_name} | 准确率(Acc): {acc:.5f} | 损失值(Loss): {loss:.5f}")

        except Exception as e:
            print(f"    [错误] 加载或测试模型 {model_name} 失败: {e}")
            results[model_name] = -1

    print("\n====== 批量测试结束 ======")
    return results


# 批量生成
def batch_run_attacks(attack_func, attack_name):
    # 定义所有需要运行的模型列表
    model_list = [
        'resnet56_cifar10',
        'sepreresnet110_cifar10',
        'seresnet20_cifar10',
        'densenet40_k12_cifar10',
        'ror3_110_cifar10',
        'diapreresnet56_cifar10'
    ]

    print(f"开始批量运行攻击: {attack_name} ...")

    for model in model_list:
        print(f"正在处理模型: {model}")
        run_attack_pipeline(
            model_name=model,
            attack_func=attack_func,
            attack_name=attack_name
        )

    print("所有模型处理完毕。")


# 批量测试
def batch_testing():
    # 1. 需要测试模型
    """
        ResNet-56   SE-PreResNet-110    SE-ResNet-20    DenseNet-BC-40 (k=12)   RoR-3-110
        DIA-PreResNet-56
    """
    target_models = [
        'resnet56_cifar10',
        'sepreresnet110_cifar10',
        'seresnet20_cifar10',
        'densenet40_k12_cifar10',
        'ror3_110_cifar10',
        'diapreresnet56_cifar10'
    ]

    # 2. 选择对抗样本
    # current_adv_dir = './data'  # 测试原始准确率

    # current_adv_dir = './diapreresnet56_cifar10_fgsm'  # fgsm

    # ifgsm
    # current_adv_dir = './resnet56_cifar10_ifgsm'
    # current_adv_dir = './sepreresnet110_cifar10_ifgsm'
    # current_adv_dir = './seresnet20_cifar10_ifgsm'
    # current_adv_dir = './densenet40_k12_cifar10_ifgsm'
    # current_adv_dir = './ror3_110_cifar10_ifgsm'
    # current_adv_dir = './diapreresnet56_cifar10_ifgsm'

    # mifgsm
    # current_adv_dir = './resnet56_cifar10_mifgsm'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm'
    # current_adv_dir = './seresnet20_cifar10_mifgsm'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm'
    # current_adv_dir = './ror3_110_cifar10_mifgsm'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm'

    # fgsm_noisy
    # current_adv_dir = './resnet56_cifar10_fgsm_noisy'
    # current_adv_dir = './sepreresnet110_cifar10_fgsm_noisy'
    # current_adv_dir = './seresnet20_cifar10_fgsm_noisy'
    # current_adv_dir = './densenet40_k12_cifar10_fgsm_noisy'
    # current_adv_dir = './ror3_110_cifar10_fgsm_noisy'
    # current_adv_dir = './diapreresnet56_cifar10_fgsm_noisy'

    # ifgsm_noisy
    # current_adv_dir = './resnet56_cifar10_ifgsm_noisy'
    # current_adv_dir = './sepreresnet110_cifar10_ifgsm_noisy'
    # current_adv_dir = './seresnet20_cifar10_ifgsm_noisy'
    # current_adv_dir = './densenet40_k12_cifar10_ifgsm_noisy'
    # current_adv_dir = './ror3_110_cifar10_ifgsm_noisy'
    # current_adv_dir = './diapreresnet56_cifar10_ifgsm_noisy'

    # mifgsm_noisy(0.1)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy'

    # mifgsm_noisy(0.2)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(0.2)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(0.2)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(0.2)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(0.2)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(0.2)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(0.2)'

    # mifgsm_noisy(0.8)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(0.8)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(0.8)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(0.8)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(0.8)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(0.8)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(0.8)'

    # mifgsm_noisy(0.3)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(0.3)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(0.3)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(0.3)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(0.3)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(0.3)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(0.3)'

    # nifgsm
    # current_adv_dir = './resnet56_cifar10_nifgsm'
    # current_adv_dir = './sepreresnet110_cifar10_nifgsm'
    # current_adv_dir = './seresnet20_cifar10_nifgsm'
    # current_adv_dir = './densenet40_k12_cifar10_nifgsm'
    # current_adv_dir = './ror3_110_cifar10_nifgsm'
    # current_adv_dir = './diapreresnet56_cifar10_nifgsm'

    # gra
    # current_adv_dir = './resnet56_cifar10_gra'
    # current_adv_dir = './sepreresnet110_cifar10_gra'
    # current_adv_dir = './seresnet20_cifar10_gra'
    # current_adv_dir = './densenet40_k12_cifar10_gra'
    # current_adv_dir = './ror3_110_cifar10_gra'
    # current_adv_dir = './diapreresnet56_cifar10_gra'

    # pgd
    # current_adv_dir = './resnet56_cifar10_pgd'
    # current_adv_dir = './sepreresnet110_cifar10_pgd'
    # current_adv_dir = './seresnet20_cifar10_pgd'
    # current_adv_dir = './densenet40_k12_cifar10_pgd'
    # current_adv_dir = './ror3_110_cifar10_pgd'
    # current_adv_dir = './diapreresnet56_cifar10_pgd'

    # mifgsm_noisy(beta=0.5,N=5)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=0.5,N=5)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=0.5,N=5)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=0.5,N=5)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=0.5,N=5)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=0.5,N=5)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=0.5,N=5)'

    # mifgsm_noisy(beta=1,N=5)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=1,N=5)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=1,N=5)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=1,N=5)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=1,N=5)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=1,N=5)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=1,N=5)'

    # mifgsm_noisy(beta=1.5,N=5)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=1.5,N=5)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=1.5,N=5)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=1.5,N=5)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=1.5,N=5)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=1.5,N=5)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=1.5,N=5)'

    # mifgsm_noisy(beta=2.0,N=5)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=2.0,N=5)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=2.0,N=5)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=2.0,N=5)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=2.0,N=5)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=2.0,N=5)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=2.0,N=5)'

    # mifgsm_noisy(beta=2.5,N=5)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=2.5,N=5)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=2.5,N=5)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=2.5,N=5)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=2.5,N=5)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=2.5,N=5)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=2.5,N=5)'

    # mifgsm_noisy(beta=3,N=5)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=3,N=5)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=3,N=5)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=3,N=5)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=3,N=5)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=3,N=5)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=3,N=5)'

    # mifgsm_noisy(beta=3.5,N=5)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=3.5,N=5)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=3.5,N=5)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=3.5,N=5)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=3.5,N=5)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=3.5,N=5)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=3.5,N=5)'

    # mifgsm_noisy(beta=4,N=5)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=4,N=5)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=4,N=5)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=4,N=5)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=4,N=5)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=4,N=5)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=4,N=5)'

    # mifgsm_noisy(beta=2.5,N=0)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=2.5,N=0)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=2.5,N=0)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=2.5,N=0)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=2.5,N=0)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=2.5,N=0)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=2.5,N=0)'

    # mifgsm_noisy(beta=2.5,N=10)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=2.5,N=10)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=2.5,N=10)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=2.5,N=10)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=2.5,N=10)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=2.5,N=10)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=2.5,N=10)'

    # mifgsm_noisy(beta=2.5,N=15)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=2.5,N=15)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=2.5,N=15)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=2.5,N=15)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=2.5,N=15)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=2.5,N=15)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=2.5,N=15)'

    # mifgsm_noisy(beta=2.5,N=20)
    # current_adv_dir = './resnet56_cifar10_mifgsm_noisy(beta=2.5,N=20)'
    # current_adv_dir = './sepreresnet110_cifar10_mifgsm_noisy(beta=2.5,N=20)'
    # current_adv_dir = './seresnet20_cifar10_mifgsm_noisy(beta=2.5,N=20)'
    # current_adv_dir = './densenet40_k12_cifar10_mifgsm_noisy(beta=2.5,N=20)'
    # current_adv_dir = './ror3_110_cifar10_mifgsm_noisy(beta=2.5,N=20)'
    # current_adv_dir = './diapreresnet56_cifar10_mifgsm_noisy(beta=2.5,N=20)'

    current_adv_dirs = [
        './resnet56_cifar10_fgsm',
        './sepreresnet110_cifar10_fgsm',
        './seresnet20_cifar10_fgsm',
        './densenet40_k12_cifar10_fgsm',
        './ror3_110_cifar10_fgsm',
        './diapreresnet56_cifar10_fgsm',

        './resnet56_cifar10_ifgsm',
        './sepreresnet110_cifar10_ifgsm',
        './seresnet20_cifar10_ifgsm',
        './densenet40_k12_cifar10_ifgsm',
        './ror3_110_cifar10_ifgsm',
        './diapreresnet56_cifar10_ifgsm',

        './resnet56_cifar10_mifgsm',
        './sepreresnet110_cifar10_mifgsm',
        './seresnet20_cifar10_mifgsm',
        './densenet40_k12_cifar10_mifgsm',
        './ror3_110_cifar10_mifgsm',
        './diapreresnet56_cifar10_mifgsm',
    ]

    # 3. 测试
    for current_adv_dir in current_adv_dirs:
        print(f"\n====== 开始测试: {current_adv_dir} ======")
        evaluate_transferability(
            adv_image_dir=current_adv_dir,
            target_model_list=target_models
        )


# ==========================================
# 集成攻击批量生成对抗样本
# ==========================================
def run_ensemble_attack_pipeline(model_names_list, ensemble_alias, attack_func, attack_name, data_root='./data'):
    """
    集成对抗样本生成流水线

    Args:
        model_names_list (list): 组成集成的模型名称列表 (e.g., ['resnet56_cifar10', 'vgg16...'])
        ensemble_alias (str): 集成模型的别名，用于文件夹命名 (e.g., 'Ensemble-6')
        attack_func (function): 攻击算法函数 (e.g., mifgsm)
        attack_name (str): 攻击算法名称 (e.g., 'mifgsm_noisy')
        data_root (str): 数据集根目录
    """

    # --- 1. 准备路径名称 ---
    # 文件夹格式通常为: 集成名_攻击方法 (例如: Ensemble-6_mifgsm)
    save_folder_name = f"{ensemble_alias}_{attack_name}"
    output_dir = os.path.join(os.path.dirname(data_root), save_folder_name)

    # 检查是否已经存在，避免重复跑
    if os.path.exists(output_dir):
        print(f"[提示] 输出目录已存在: {output_dir}，跳过或请手动删除。")
        # return # 如果不想覆盖可以取消注释

    # --- 2. 加载数据 ---
    print("1. [集成] 正在加载数据...")
    adv_set = AdvDataset(data_root, transform=transform)
    adv_names = adv_set.__getname__()
    adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)

    # --- 3. 加载集成模型 ---
    print(f"2. [集成] 正在构建集成模型 (包含 {len(model_names_list)} 个子模型)...")
    print(f"   模型列表: {model_names_list}")

    # 初始化你的 ensembleNet (确保 ensembleNet 类已定义)
    model = ensembleNet(model_names_list).to(device)
    model.eval()  # 必须开启 eval 模式

    # 为了防止显存溢出 (OOM)，如果模型很多，可以考虑在此处 torch.no_grad() 上下文外层包裹
    # 攻击需要反向传播，所以 gen_adv_examples 内部必须允许梯度

    loss_fn = nn.CrossEntropyLoss()

    # --- 4. 生成对抗样本 ---
    print(f"3. [集成] 正在执行攻击算法: {attack_name} ...")

    # 将集成模型当做一个普通模型传入
    adv_examples = gen_adv_examples(model, adv_loader, attack_func, loss_fn)

    # --- 5. 保存结果 ---
    print(f"4. [集成] 正在保存结果到: {save_folder_name}")
    create_dir(data_root, output_dir, adv_examples, adv_names)
    print("====== 集成攻击生成完毕 ======\n")


# ==========================================
# 批量运行集成攻击
# ==========================================
def batch_run_ensemble_attacks(attack_func, attack_name):
    """
    配置并运行集成攻击
    """

    # 白盒模型列表
    all_models = [
        'resnet56_cifar10',
        'sepreresnet110_cifar10',
        'seresnet20_cifar10',
        'densenet40_k12_cifar10',
        'ror3_110_cifar10',
        'diapreresnet56_cifar10'
    ]

    print(f"开始批量运行集成攻击: {attack_name}")

    # --- 场景 使用所有 6 个模型进行集成 ---
    """
    # 文件夹名结果示例: Ensemble-All_mifgsm
    run_ensemble_attack_pipeline(
        model_names_list=all_models,
        ensemble_alias="Ensemble-All",  # 自定义文件夹前缀
        attack_func=attack_func,
        attack_name=attack_name
    )
    """

    # --- 留一法 生成(5个模型集成，攻击第6个) ---
    for hold_out_model in all_models:
        # 创建一个不包含当前 hold_out_model 的列表
        source_models = [m for m in all_models if m != hold_out_model]

        # 命名示例: Ensemble-Minus-ResNet56
        # 我们可以简化命名，比如 "Ens-Holdout-0", "Ens-Holdout-1"
        alias = f"Ens-Holdout-{hold_out_model.split('_')[0]}"

        print(f"正在处理 Hold-out 集成: 排除 {hold_out_model}")

        run_ensemble_attack_pipeline(
            model_names_list=source_models,
            ensemble_alias=alias,
            attack_func=attack_func,
            attack_name=attack_name
        )

    print("所有集成任务处理完毕。")


# 测试集成样本的迁移性
def batch_testing_ensemble():

    target_models = [
        'resnet56_cifar10',
        'sepreresnet110_cifar10',
        'seresnet20_cifar10',
        'densenet40_k12_cifar10',
        'ror3_110_cifar10',
        'diapreresnet56_cifar10'
    ]

    # 需要测试的集成样本目录
    # ensemble_adv_dir = './Ens-Holdout-resnet56_mifgsm'
    # ensemble_adv_dir = 'Ens-Holdout-sepreresnet110_mifgsm'
    # ensemble_adv_dir = 'Ens-Holdout-seresnet20_mifgsm'
    # ensemble_adv_dir = './Ens-Holdout-densenet40_mifgsm'
    # ensemble_adv_dir = 'Ens-Holdout-ror3_mifgsm'
    # ensemble_adv_dir = 'Ens-Holdout-diapreresnet56_mifgsm'

    # ensemble_adv_dir = './Ens-Holdout-resnet56_mifgsm_noisy(0.1)'
    # ensemble_adv_dir = 'Ens-Holdout-sepreresnet110_mifgsm_noisy(0.1)'
    # ensemble_adv_dir = 'Ens-Holdout-seresnet20_mifgsm_noisy(0.1)'
    # ensemble_adv_dir = './Ens-Holdout-densenet40_mifgsm_noisy(0.1)'
    # ensemble_adv_dir = 'Ens-Holdout-ror3_mifgsm_noisy(0.1)'
    # ensemble_adv_dir = 'Ens-Holdout-diapreresnet56_mifgsm_noisy(0.1)'

    # ensemble_adv_dir = './Ens-Holdout-resnet56_mifgsm_noisy(0.2)'
    # ensemble_adv_dir = 'Ens-Holdout-sepreresnet110_mifgsm_noisy(0.2)'
    # ensemble_adv_dir = 'Ens-Holdout-seresnet20_mifgsm_noisy(0.2)'
    # ensemble_adv_dir = './Ens-Holdout-densenet40_mifgsm_noisy(0.2)'
    # ensemble_adv_dir = 'Ens-Holdout-ror3_mifgsm_noisy(0.2)'
    ensemble_adv_dir = 'Ens-Holdout-diapreresnet56_mifgsm_noisy(0.2)'

    # ensemble_adv_dir = './Ens-Holdout-resnet56_mifgsm_noisy(0.3)'
    # ensemble_adv_dir = 'Ens-Holdout-sepreresnet110_mifgsm_noisy(0.3)'
    # ensemble_adv_dir = 'Ens-Holdout-seresnet20_mifgsm_noisy(0.3)'
    # ensemble_adv_dir = './Ens-Holdout-densenet40_mifgsm_noisy(0.3)'
    # ensemble_adv_dir = 'Ens-Holdout-ror3_mifgsm_noisy(0.3)'
    # ensemble_adv_dir = 'Ens-Holdout-diapreresnet56_mifgsm_noisy(0.3)'

    # ensemble_adv_dir = './Ens-Holdout-resnet56_mifgsm_noisy(0.4)'
    # ensemble_adv_dir = 'Ens-Holdout-sepreresnet110_mifgsm_noisy(0.4)'
    # ensemble_adv_dir = 'Ens-Holdout-seresnet20_mifgsm_noisy(0.4)'
    # ensemble_adv_dir = './Ens-Holdout-densenet40_mifgsm_noisy(0.4)'
    # ensemble_adv_dir = 'Ens-Holdout-ror3_mifgsm_noisy(0.4)'
    # ensemble_adv_dir = 'Ens-Holdout-diapreresnet56_mifgsm_noisy(0.4)'

    evaluate_transferability(
        adv_image_dir=ensemble_adv_dir,
        target_model_list=target_models
    )


if __name__ == '__main__':

    # --- 批量生成对抗样本 ---
    # batch_run_attacks(fgsm, 'fgsm')  # fgsm
    # batch_run_attacks(ifgsm, 'ifgsm')  # ifgsm
    # batch_run_attacks(mifgsm, 'mifgsm')  # mifgsm
    # batch_run_attacks(fgsm_noisy, 'fgsm_noisy')  # fgsm-noisy
    # batch_run_attacks(ifgsm_noisy, 'ifgsm_noisy')  # ifgsm-noisy
    # batch_run_attacks(mifgsm_noisy, 'mifgsm_noisy')  # mifgsm-noisy
    # batch_run_attacks(mifgsm_noisy, 'mifgsm_noisy(0.2)')  # mifgsm-noisy(0.2)
    # batch_run_attacks(mifgsm_noisy, 'mifgsm_noisy(0.8)')  # mifgsm-noisy(0.8)
    # batch_run_attacks(mifgsm_noisy, 'mifgsm_noisy(0.3)')  # mifgsm-noisy(0.3)
    # batch_run_attacks(nifgsm, 'nifgsm')  # nifgsm
    # batch_run_attacks(gra, 'gra')  # gra
    # batch_run_attacks(pgd, 'pgd')  # pgd
    # batch_run_attacks(vmi_fgsm, 'vmi_fgsm')  # vmi_fgsm

    # noisy_attack_func = partial(mifgsm_noisy, beta=1, N=5)
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=1,N=5)')

    # noisy_attack_func = partial(mifgsm_noisy, beta=2.5, N=5)
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=1.5,N=5)')
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=2.0,N=5)')
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=2.5,N=5)')

    # noisy_attack_func = partial(mifgsm_noisy, beta=3, N=5)
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=3,N=5)')

    # noisy_attack_func = partial(mifgsm_noisy, beta=3.5, N=5)
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=3.5,N=5)')

    # noisy_attack_func = partial(mifgsm_noisy, beta=4, N=5)
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=4,N=5)')

    # noisy_attack_func = partial(mifgsm_noisy, beta=2.5, N=0)
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=2.5,N=0)')

    # noisy_attack_func = partial(mifgsm_noisy, beta=2.5, N=10)
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=2.5,N=10)')

    # noisy_attack_func = partial(mifgsm_noisy, beta=2.5, N=15)
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=2.5,N=15)')

    # noisy_attack_func = partial(mifgsm_noisy, beta=2.5, N=20)
    # batch_run_attacks(noisy_attack_func, 'mifgsm_noisy(beta=2.5,N=20)')

    # --- 批量测试 ---
    batch_testing()

    # --- 批量生成集成对抗样本 ---
    # noisy_attack_func = partial(mifgsm_noisy, noise_scale=0.2)
    # batch_run_ensemble_attacks(noisy_attack_func, "mifgsm_noisy(0.1)")
    # batch_run_ensemble_attacks(noisy_attack_func, "mifgsm_noisy(0.2)")
    # batch_run_ensemble_attacks(noisy_attack_func, "mifgsm_noisy(0.3)")
    # batch_run_ensemble_attacks(noisy_attack_func, "mifgsm_noisy(0.4)")
    # batch_run_ensemble_attacks(mifgsm, "mifgsm")

    # --- 批量测试(集成测试) ---
    # batch_testing_ensemble()
