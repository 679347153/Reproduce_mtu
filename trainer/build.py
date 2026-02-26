"""
================================================================================
                          训练器构建与基础训练框架
================================================================================

文件概述：
    此模块负责构建训练器并提供训练过程的基础实现，包含：
    1. `Tracker`：用于记录实验进度（epoch、best_result 等）
    2. `BaseTrainer`：训练器基类，实现训练/验证/测试流程的通用逻辑
    3. `build_trainer(cfg)`：从注册表中实例化特定 Trainer 的工厂函数

核心职责：
    - 初始化分布式训练的 Accelerator（gradient accumulation、ddp 等）
    - 构建数据加载器、模型、优化器/调度器、评估器
    - 管理检查点加载/保存、断点恢复和预训练权重加载
    - 提供统一的日志记录接口（集成 wandb 与 accelerator.log）

设计要点：
    - 通过 `TRAINER_REGISTRY` 注册不同训练器实现，以支持多个训练阶段或策略
    - 将分布式与单机逻辑抽象在 Accelerator 中，BaseTrainer 仅关注训练流程
    - 使用 omegaconf 管理配置并在初始化时解析随机种子与训练超参

适用场景：
    MTU3D 的训练入口（`run.py`）会调用 `build_trainer(cfg)` 来创建并运行训练器。

================================================================================
"""

import copy as cp
import glob
import os
from datetime import timedelta
from pathlib import Path
from omegaconf import OmegaConf
from omegaconf import open_dict
from tqdm import tqdm
import numpy as np

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from fvcore.common.registry import Registry
import torch
import wandb

import common.io_utils as iu
from common.io_utils import make_dir
import common.misc as misc
from data.build import build_dataloader
from evaluator.build import build_eval
from model.build import build_model
from optim.build import build_optim


TRAINER_REGISTRY = Registry("Trainer")


class Tracker():
    """实验追踪器（轻量级）：记录当前 epoch、实验名称和历史最佳结果。"""
    def __init__(self, cfg):
        # 使用配置初始化追踪器状态
        self.reset(cfg)

    def step(self):
        self.epoch += 1

    def reset(self, cfg):
        self.exp_name = f"{cfg.exp_dir.parent.name.replace(f'{cfg.name}', '').lstrip('_')}/{cfg.exp_dir.name}"
        self.epoch = 0
        self.best_result = -np.inf

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

@TRAINER_REGISTRY.register()
class BaseTrainer():
    def __init__(self, cfg):
        """初始化 BaseTrainer。

        主要职责：
            - 根据 cfg 初始化随机种子、调试开关
            - 初始化 Accelerator（用于分布式训练与梯度累积）
            - 构建数据加载器、模型、优化器、评估器
            - 准备模型与优化器用于分布式训练（accelerator.prepare）
            - 支持加载预训练权重与断点恢复

        参数:
            cfg: OmegaConf 配置对象
        """
        # 固定随机种子以保证可复现性
        set_seed(cfg.rng_seed)
        # 调试标志（用于控制日志/可视化行为）
        self.debug = cfg.debug.get("flag", False)
        self.hard_debug = cfg.debug.get("hard_debug", False)
        # 验证与保存周期配置
        self.epochs_per_eval = cfg.solver.get("epochs_per_eval", None)
        self.epochs_per_save = cfg.solver.get("epochs_per_save", None)
        self.global_step = 0

        # 初始化实验追踪器（记录 epoch、best_result 等）
        self.exp_tracker = Tracker(cfg)

        # 配置 WandB 参数（仅用于 accelerator.init_trackers 时传入）
        wandb_args = {"entity": cfg.logger.entity, "id": cfg.logger.run_id, "resume": cfg.resume}
        if not cfg.logger.get('autoname'):
            wandb_args["name"] = self.exp_tracker.exp_name
        wandb_args['mode'] = 'offline'

        # 全局 logger（accelerate 的 logger）
        self.logger = get_logger(__name__)
        self.mode = cfg.mode

        # DDP 与进程组初始化参数（用于 accelerate）
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=216000))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        use_ddp = cfg.num_gpu > 1 or world_size > 1
        kwargs = ([ddp_kwargs] if use_ddp else []) + [init_kwargs]

        # 初始化 Accelerator（支持梯度累积与日志记录）
        gradient_accumulation_steps = cfg.solver.get("gradient_accumulation_steps", 1)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs
        )

        # 初始化 tracker 与 wandb（在非硬调试模式下）
        if not self.hard_debug:
            self.accelerator.init_trackers(
                    project_name=cfg.name if not self.debug else "Debug",
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) if not cfg.resume else None,
                    init_kwargs={"wandb": wandb_args}
                )
        # 打印配置以便调试与复现
        print(OmegaConf.to_yaml(cfg))

        # 构建数据加载器、模型与优化器/调度器
        keys = ["train", "val", "test"] if self.mode == "train" else ["test"]
        self.data_loaders = {key : build_dataloader(cfg, split=key) for key in keys}
        self.model = build_model(cfg) # ！！！！！！！！！构建模型！！！！！！！！！

        """
                                        构建模型!!!!!!
        """
        if self.mode == "test":
            total_steps = 1
        else:
            total_steps = (len(self.data_loaders["train"]) * cfg.solver.epochs) // gradient_accumulation_steps
        self.loss, self.optimizer, self.scheduler = build_optim(cfg, self.model.get_opt_params(),
                                                                total_steps= total_steps)

        # 构建评估器（可选传入 dataloaders）
        if misc.rgetattr(cfg, "eval.pass_kwargs", False):
            kwargs = {"dataloaders": self.data_loaders}
        else:
            kwargs = {}
        self.evaluator = build_eval(cfg, self.accelerator, **kwargs)

        # 训练参数归档
        self.epochs = cfg.solver.epochs
        self.total_steps = 1 if self.mode == "test" else len(self.data_loaders["train"]) * cfg.solver.epochs
        self.grad_norm = cfg.solver.get("grad_norm")

        # 加载预训练权重（若提供路径）
        if cfg.get('pretrain_ckpt_path'):
            self.pretrain_ckpt_path = Path(cfg.pretrain_ckpt_path)
            self.load_pretrain()

        # 使用 Accelerator 准备模型、损失与优化器（处理分布式包装）
        self.model, self.loss, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.loss, self.optimizer, self.scheduler)
        for name, loader in self.data_loaders.items():
            if isinstance(loader, list):
                loader = self.accelerator.prepare(*loader)
            else:
                loader = self.accelerator.prepare(loader)
            self.data_loaders[name] = loader
        # 将追踪器注册到 accelerator，以便 checkpoint 时保存状态
        self.accelerator.register_for_checkpointing(self.exp_tracker)

        # 检查断点恢复配置
        self.ckpt_path = Path(cfg.ckpt_path) if cfg.get("ckpt_path") else Path(cfg.exp_dir) / "ckpt" / "latest.pth"
        if cfg.resume:
            self.resume()

    def forward(self, data_dict):
        return self.model(data_dict)

    def backward(self, loss):
        # Need to be reimplemented when using different sets of optimizer and schedulers
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def log(self, results, mode="train"):
        if not self.hard_debug:
            log_dict = {}
            for key, val in results.items():
                val = val.item() if isinstance(val, torch.Tensor) else val
                log_dict[f"{mode}/{key}"] = val
            if mode == "train":
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_name = param_group.get("name", "")
                    param_name = param_name if len(param_name) > 0 else i
                    log_dict[f"{mode}/lr/{param_name}"] = param_group['lr']
            self.accelerator.log(log_dict, step=self.global_step)

    def save(self, name):
        make_dir(self.ckpt_path.parent)
        self.save_func(str(self.ckpt_path.parent / name))

    def resume(self):
        if self.ckpt_path.exists():
            print(f"Resuming from {str(self.ckpt_path)}")
            # self.logger.info(f"Resuming from {str(self.ckpt_path)}")
            self.accelerator.load_state(str(self.ckpt_path))
            # self.logger.info(f"Successfully resumed from {self.ckpt_path}")
            print(f"Successfully resumed from {self.ckpt_path}")
        else:
            self.logger.info("training from scratch")

    def load_pretrain(self):
        print(f"Loading pretrained weights from {str(self.pretrain_ckpt_path)}")
        model_weight_path_pattern = str(self.pretrain_ckpt_path / "pytorch_model*.bin")
        model_weight_paths = glob.glob(model_weight_path_pattern)
        if len(model_weight_paths) == 0:
            raise FileNotFoundError(f"Cannot find pytorch_model.bin in {str(self.pretrain_ckpt_path)}")
        weights = {}
        for model_weight_path in model_weight_paths:
            weights.update(torch.load(model_weight_path, map_location="cpu"))
        warning = self.model.load_state_dict(weights, strict=False)
        print(f"Successfully loaded from {str(self.pretrain_ckpt_path)}: {warning}")

    def save_func(self, path):
        self.accelerator.save_state(path, safe_serialization=False)
    

def build_trainer(cfg):
    return TRAINER_REGISTRY.get(cfg.trainer)(cfg)
