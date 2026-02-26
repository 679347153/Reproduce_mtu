"""
================================================================================
                    MTU3D 阶段一训练器（EmbodiedStage1Trainer）
================================================================================

文件概述：
    这是 MTU3D 项目的实体感知第一阶段训练器，负责处理多模态 3D 场景理解任务的训练流程。
    支持以下核心功能：
    1. 多轮 epoch 训练，支持中断恢复
    2. 梯度累积和分布式训练（通过 Accelerator）
    3. 定期验证和模型检查点保存
    4. 测试阶段的推理和评估
    5. WandB 日志记录和指标追踪

核心架构：
    - 继承自 BaseTrainer：基础训练框架（包含优化器、调度器、加速器等）
    - 前向传播：data_dict → model → output_dict
    - 反向传播：loss → 梯度清零 → backward pass → 梯度裁剪 → 优化器步 → 调度器步
    - 训练循环：train_step → eval_step（定期）→ test_step
    - 检查点管理：保存 best.pth（最佳指标）、ckpt_epoch.pth（周期保存）、latest.pth（最新模型）

主要工作流程：
    1. 初始化：加载配置、模型、优化器、数据加载器
    2. 训练阶段：for epoch in range(start_epoch, epochs):
       - train_step：遍历训练数据，计算损失，反向传播
       - 定期 eval_step：在验证集上评估，记录指标
       - 检查点保存：保存最佳和周期检查点
    3. 测试阶段：test_step 在测试集上进行最终评估
    4. 结束：end_training（清理资源）

关键功能特性：
    - 梯度累积：支持在多个批次上累积梯度
    - 梯度裁剪：防止梯度爆炸
    - 分布式训练：支持多 GPU/多节点（通过 accelerator）
    - 最佳检查点跟踪：比较 target_metric 保存最佳模型
    - 灵活的评估频率：通过 epochs_per_eval 控制
    - 时间统计：记录每个 epoch 的训练时间和剩余时间
    - 时间测量：eval/test 阶段记录推理耗时（分钟）

继承和注册：
    - @TRAINER_REGISTRY.register()：将此类注册到训练器工厂
    - super().__init__(cfg)：初始化 BaseTrainer 的所有组件

配置参数（来自 cfg）：
    - eval.compute_loss_eval：在验证/测试时是否计算损失
    - model, optimizer, scheduler, loss：由 BaseTrainer 初始化
    - data_loaders：train/val/test 数据加载器
    - epochs, epochs_per_eval, epochs_per_save：训练超参
    - grad_norm：梯度裁剪阈值（None 表示禁用）

================================================================================
"""

from time import time
from tqdm import tqdm

import torch
from trainer.build import TRAINER_REGISTRY
from trainer.build import BaseTrainer


@TRAINER_REGISTRY.register()
class EmbodiedStage1Trainer(BaseTrainer):
    """阶段一训练器：管理多轮 epoch 的训练、验证和测试流程。
    
    该类实现了标准的 PyTorch 训练循环，集成了分布式训练、梯度累积、检查点管理
    等高级功能。继承自 BaseTrainer，复用优化器、调度器、加速器等核心组件。
    """
    
    def __init__(self, cfg):
        """初始化训练器。
        
        参数:
            cfg: OmegaConf 配置对象，包含所有训练超参和模型配置
        """
        super().__init__(cfg)
        # 控制是否在验证/测试阶段计算损失值（通常为 False，节省计算）
        self.compute_loss_eval = cfg.eval.get("compute_loss_eval", False)

    def forward(self, data_dict):
        """前向传播：将数据传入模型进行推理。
        
        参数:
            data_dict (dict): 包含输入数据的字典（点云、图像、文本等）
        
        返回:
            dict: 模型的输出字典（包含预测、特征等）
        """
        return self.model(data_dict)

    def backward(self, loss):
        """反向传播和优化器更新。
        
        执行流程：
            1. 清零梯度（optimizer.zero_grad）
            2. 计算反向传播（accelerator.backward）
            3. 梯度裁剪（防止梯度爆炸）
            4. 优化器步（更新参数）
            5. 调度器步（调整学习率）
        
        参数:
            loss: 标量损失张量
        
        返回:
            dict: 包含梯度范数等统计信息的字典
        
        注:
            如果使用多个优化器/调度器，需要重新实现此方法
        """
        backward_dict = {}
        
        # 清零梯度，准备进行反向传播
        self.optimizer.zero_grad()
        
        # 使用 Accelerator 的 backward 方法处理分布式训练中的梯度同步
        self.accelerator.backward(loss)
        
        # 梯度裁剪：防止梯度爆炸（仅在分布式同步时执行）
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            cur_grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            # 记录实际的梯度范数用于监控
            backward_dict.update({"grad_norm": cur_grad_norm})
        
        # 优化器步：更新模型参数
        self.optimizer.step()
        
        # 调度器步：根据训练进度调整学习率
        self.scheduler.step()
        
        return backward_dict

    def train_step(self, epoch):
        """执行一个 epoch 的训练。
        
        流程:
            1. 设置模型为训练模式
            2. 遍历训练数据加载器
            3. 对每个批次执行：前向传播 → 计算损失 → 反向传播 → 日志记录
            4. 支持梯度累积（通过 accelerator.accumulate）
        
        参数:
            epoch (int): 当前 epoch 索引（从 0 开始）
        """
        # 设置模型为训练模式（启用 dropout/batchnorm 等）
        self.model.train()
        
        # 获取训练数据加载器
        loader = self.data_loaders["train"]
        
        # 创建进度条（仅在主进程显示）
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process), 
                   desc=f"[Epoch {epoch + 1}/{self.epochs}]")
        
        # 遍历训练批次
        for i, data_dict in enumerate(loader):
            # 梯度累积：在多个批次上累积梯度后再更新
            with self.accelerator.accumulate(self.model):
                # 前向传播：获取模型输出
                data_dict = self.forward(data_dict)
                
                # 计算损失：返回总损失和各任务的损失详情
                loss, losses = self.loss(data_dict)
                
                # 反向传播和优化器更新
                backward_dict = self.backward(loss)
                
                # 更新全局步数（用于学习率调度和日志）
                self.global_step += 1
                
                # 整合日志信息：步数 + 各任务损失 + 梯度信息
                log_dict = {'step': self.global_step}
                log_dict.update(losses)  # 任务损失（mask_loss, openvocab_loss 等）
                log_dict.update(backward_dict)  # 梯度信息（grad_norm）
                
                # 记录到 WandB 和日志系统
                self.log(log_dict, mode="train")
                
                # 更新进度条
                pbar.update(1)

    @torch.no_grad()
    def eval_step(self, epoch):
        """在验证集上评估模型性能。
        
        流程:
            1. 设置模型为评估模式（禁用 dropout/batchnorm）
            2. 遍历验证数据，前向传播
            3. 收集预测结果用于评估
            4. 计算评估指标（精度/IoU/mAP 等）
            5. 与之前最佳指标比较，判断是否保存
            6. 记录时间和指标到日志
        
        参数:
            epoch (int): 当前 epoch 索引
        
        返回:
            bool: 是否为目前最佳模型
        """
        # 设置模型为评估模式（禁用 dropout/batchnorm 等的随机性）
        self.model.eval()
        
        # 获取验证数据加载器
        loader = self.data_loaders["val"]
        
        # 创建进度条（仅在主进程显示）
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        
        # 记录评估开始时间
        st = time()
        
        # 遍历验证批次（禁用梯度计算以节省内存）
        for i, data_dict in enumerate(loader):
            # 前向传播：获取模型预测
            data_dict = self.forward(data_dict)
            
            # 可选：计算验证损失（通常不需要，配置可控）
            if self.compute_loss_eval:
                loss, losses = self.loss(data_dict)
            
            # 更新评估器：累积预测结果和真值用于计算指标
            self.evaluator.update(data_dict)
            
            # 更新进度条
            pbar.update(1)
        
        # 计算评估指标：返回是否为最佳（初步）和详细结果字典
        is_best, results = self.evaluator.record()
        
        # 计算推理耗时（分钟）
        end = time()
        results['time'] = (end - st) / 60
        
        # 记录评估结果到日志系统（WandB 等）
        self.log(results, mode="val")
        
        # 重置评估器，准备下一次评估
        self.evaluator.reset()
        
        # 与历史最佳指标比较，决定是否更新最佳模型记录
        target_metric = results['target_metric']  # 主要评估指标（如 mAP）
        if target_metric > self.exp_tracker.best_result:
            # 新的最佳指标，更新记录
            is_best = True
            self.exp_tracker.best_result = target_metric
        else:
            # 未达到最佳，不更新
            is_best = False
        
        return is_best

    @torch.no_grad()
    def test_step(self):
        """在测试集上进行最终评估。
        
        流程:
            1. 设置模型为评估模式
            2. 遍历测试数据，前向传播
            3. 收集预测结果
            4. 计算最终评估指标
            5. 记录结果（通常保存到文件用于进一步分析）
        
        返回:
            dict: 包含所有测试指标的结果字典
        """
        # 设置模型为评估模式
        self.model.eval()
        
        # 获取测试数据加载器
        loader = self.data_loaders["test"]
        
        # 创建进度条（仅在主进程显示）
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        
        # 遍历测试批次（禁用梯度计算）
        for i, data_dict in enumerate(loader):
            # 前向传播：获取模型预测
            data_dict = self.forward(data_dict)
            
            # 可选：计算测试损失
            if self.compute_loss_eval:
                loss, losses = self.loss(data_dict)
            
            # 更新评估器：累积预测和真值
            self.evaluator.update(data_dict)
            
            # 更新进度条
            pbar.update(1)
        
        # 计算测试指标
        is_best, results = self.evaluator.record()
        
        # 记录测试结果到日志系统
        self.log(results, mode="test")
        
        # 重置评估器
        self.evaluator.reset()
        
        return results

    def run(self):
        """主训练循环：协调训练、验证、测试的整个流程。
        
        训练工作流（当 mode="train" 时）:
            1. 初始化起始 epoch（支持断点恢复）
            2. 多轮 epoch 循环：
               - 执行 train_step：一个 epoch 的训练
               - 定期执行 eval_step：验证模型性能
               - 定期保存检查点（best/epoch/latest）
            3. 训练完成后执行 test_step
            4. 清理资源（end_training）
        
        测试工作流（当 mode="test" 时）:
            仅执行 test_step，获取最终评估结果
        """
        if self.mode == "train":
            # ========== 初始化训练参数 ==========
            # 从实验追踪器获取起始 epoch（支持断点恢复）
            start_epoch = self.exp_tracker.epoch
            
            # 初始化全局步数：基于起始 epoch 和每 epoch 的批次数
            self.global_step = start_epoch * len(self.data_loaders["train"])
            
            # ========== 多轮 epoch 训练循环 ==========
            for epoch in range(start_epoch, self.epochs):
                # 增加实验追踪器的 epoch 计数
                self.exp_tracker.step()
                
                # 记录 epoch 开始时间
                st = time()
                
                # 执行一个 epoch 的训练
                self.train_step(epoch)
                
                # 计算该 epoch 的训练耗时（分钟）
                epoch_time = (time() - st) / 60
                
                # 记录时间统计：epoch 耗时、epoch 号、剩余时间预估
                remaining_time = epoch_time * (self.epochs - epoch) / 60
                self.log({
                    "epoch_time": epoch_time,
                    "epoch": epoch + 1,
                    "remaining_time": remaining_time
                }, mode="train")

                # ========== 定期验证和检查点保存 ==========
                # 按配置的频率执行验证（epochs_per_eval=N 表示每 N 个 epoch 验证一次）
                if self.epochs_per_eval and (epoch + 1) % self.epochs_per_eval == 0:
                    # 执行验证步，返回是否为最佳模型
                    is_best = self.eval_step(epoch)
                    self.accelerator.print(f"[Epoch {epoch + 1}/{self.epochs}] finished eval, is_best: {is_best}")
                else:
                    # 未到验证周期，置 is_best=False
                    is_best = False

                # 等待所有进程完成（分布式训练同步）
                self.accelerator.wait_for_everyone()
                
                # 仅在主进程进行检查点保存（避免多进程重复保存）
                if self.accelerator.is_main_process:
                    # 如果是最佳模型，保存到 best.pth
                    if is_best:
                        self.save("best.pth")
                    
                    # 按周期保存检查点（用于恢复或分析训练过程）
                    if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                        self.save(f"ckpt_epoch.pth")
                    
                    # 总是保存最新的模型（覆盖之前的 latest.pth）
                    self.save("latest.pth")

        # ========== 最终测试阶段 ==========
        # 无论训练还是测试模式，都执行测试步骤
        self.test_step()
        
        # 训练完成后的资源清理
        if self.mode == "train":
            # 结束分布式训练环境（清理 GPU/通信资源）
            self.accelerator.end_training()
