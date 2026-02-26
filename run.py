"""
================================================================================
                          MTU3D 主训练脚本（run.py）
================================================================================

文件概述：
    这是 MTU3D 项目的主入口脚本，负责：
    1. 加载和处理配置文件（通过 Hydra 框架）
    2. 支持实验恢复（resume）功能
    3. 生成实验名称和管理实验目录
    4. 初始化训练器并启动训练流程

核心架构：
    - Hydra 配置管理：从 configs/default 读取配置，支持命令行覆盖
    - 实验追踪：集成 Weights & Biases (WandB) 用于结果追踪
    - 模块化设计：通过 trainer.build 动态构建不同的训练器
    - 异常处理：支持中断恢复并保存配置

主要工作流程：
    1. 解析配置（支持恢复已有实验或创建新实验）
    2. 生成实验名称和目录结构
    3. 保存配置文件到实验目录
    4. 构建训练器并执行训练

关键功能：
    - 支持恢复训练：检查 cfg.exp_dir 是否已存在
    - 智能命名：根据 naming_keywords 自动组合实验名称
    - WandB 集成：为每个实验分配唯一的 run_id
    - Debug 模式：用于快速验证和测试

================================================================================
"""

# 标准库导入
from pathlib import Path  # 路径处理
import hydra  # 配置管理框架
from datetime import datetime  # 时间戳生成
from omegaconf import OmegaConf, open_dict  # 配置加载和修改

# 第三方库导入
import wandb  # 实验追踪和可视化

# 项目内部导入
import common.io_utils as iu  # I/O 工具函数
from common.misc import rgetattr  # 递归获取嵌套配置属性
from trainer.build import build_trainer  # 训练器构建工厂


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg):
    """
    主函数：控制实验的整个生命周期。
    
    参数:
        cfg: Hydra 框架传入的配置对象 (DictConfig)
    
    流程:
        1. 尝试恢复已有实验或初始化新实验
        2. 生成实验名称（基于配置和命名规则）
        3. 创建实验目录并保存配置
        4. 构建并运行训练器
    """
    
    # ========== 实验恢复或初始化 ==========
    if cfg.resume:
        # 恢复模式：加载已有实验的配置
        assert Path(cfg.exp_dir).exists(), f"Resuming failed: {cfg.exp_dir} does not exist."
        print(f"Resuming from {cfg.exp_dir}")
        # 从保存的 config.yaml 重新加载配置（确保路径和超参一致）
        cfg = OmegaConf.load(Path(cfg.exp_dir) / 'config.yaml')
        cfg.resume = True
    else:
        # 新实验模式：生成唯一的 run_id 用于 WandB 追踪
        run_id = wandb.util.generate_id()
        with open_dict(cfg):
            cfg.logger.run_id = run_id
    
    # ========== 解析配置并生成实验名称 ==========
    # 解析配置中的所有变量引用（例如 ${model.hidden_size}）
    OmegaConf.resolve(cfg)
    
    # 初始化命名键列表，首个元素为实验基础名称
    naming_keys = [cfg.name]
    
    # 根据 naming_keywords 策略动态添加标识符
    for name in cfg.get('naming_keywords', []):
        if name == "time":
            # "time" 键在后续添加时间戳，这里跳过
            continue
        elif name == "task":
            # 添加任务类型和数据集信息
            naming_keys.append(cfg.task)
            if rgetattr(cfg, "data.note", None) is not None:
                # 优先使用自定义 note
                naming_keys.append(rgetattr(cfg, "data.note"))
            else:
                # 否则使用训练数据集名称列表
                datasets = rgetattr(cfg, "data.train")
                dataset_names = "+".join([str(x) for x in datasets])
                naming_keys.append(dataset_names)
        elif name == "dataloader.batchsize":
            # 添加总批大小（批大小 × GPU 数）
            naming_keys.append(f"b{rgetattr(cfg, name) * rgetattr(cfg, 'num_gpu')}")
        else:
            # 添加其他自定义配置字段（如果不为空）
            if str(rgetattr(cfg, name)) != "":
                naming_keys.append(str(rgetattr(cfg, name)))
    
    # 将所有命名键用下划线连接成实验名称
    exp_name = "_".join(naming_keys)

    # 调试模式：使用简单名称便于识别
    if rgetattr(cfg, "debug.flag", False):
        exp_name = "Debug_test"
    print(exp_name)

    # ========== 创建实验目录和保存配置 ==========
    if not cfg.exp_dir:
        # 根据实验名称和当前时间戳创建目录路径
        cfg.exp_dir = Path(cfg.base_dir) / exp_name / f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')}" 
    else:
        # 使用已指定的实验目录
        cfg.exp_dir = Path(cfg.exp_dir)
    
    # 创建实验目录（包括所有中间目录）
    iu.make_dir(cfg.exp_dir)
    
    # 将完整配置保存到 YAML 文件（便于恢复和复现）
    OmegaConf.save(cfg, cfg.exp_dir / "config.yaml")

    # ========== 构建训练器并开始训练 ==========
    # 根据 cfg.model.name 动态选择对应的训练器
    trainer = build_trainer(cfg)
    # 执行训练
    trainer.run()


if __name__ == "__main__":
    # 脚本入口点
    main()
