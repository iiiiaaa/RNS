import os

import torch.nn as nn
from torch.utils.data import DataLoader
from noise import device, ptcv_get_model, AdvDataset, transform, \
    fgsm, ifgsm, mifgsm, mifgsm_noisy, epoch_benign, gen_adv_examples, create_dir, batch_size, \
    nifgsm, gra, pgd, ti, vmi
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

        './resnet56_cifar10_ti',
        './sepreresnet110_cifar10_ti',
        './seresnet20_cifar10_ti',
        './densenet40_k12_cifar10_ti',
        './ror3_110_cifar10_ti',
        './diapreresnet56_cifar10_ti',

        './resnet56_cifar10_vmi',
        './sepreresnet110_cifar10_vmi',
        './seresnet20_cifar10_vmi',
        './densenet40_k12_cifar10_vmi',
        './ror3_110_cifar10_vmi',
        './diapreresnet56_cifar10_vmi',
    ]

    # 3. 测试
    for current_adv_dir in current_adv_dirs:
        print(f"\n====== 开始测试: {current_adv_dir} ======")
        evaluate_transferability(
            adv_image_dir=current_adv_dir,
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
    # batch_run_attacks(ti, 'ti')  # ti
    # batch_run_attacks(pgd, 'pgd')  # pgd
    # batch_run_attacks(vmi, 'vmi')  # vmi

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


