from time import time
import numpy as np
from tqdm import tqdm

import torch
from trainer.build import TRAINER_REGISTRY
from trainer.build import BaseTrainer


@TRAINER_REGISTRY.register()
class EmbodiedStage2Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        # convert loaders and evaluators to list
        for split in ['val', 'test']:
            if split not in self.data_loaders:
                continue
            loaders = self.data_loaders[split]
            if not (isinstance(loaders, list) or isinstance(loaders, tuple)):
                self.data_loaders[split] = [loaders]
        if not isinstance(self.evaluator, list):
            self.evaluator = [self.evaluator]

    def forward(self, data_dict):
        return self.model(data_dict)

    def backward(self, loss):
        backward_dict = {}
        # Need to be reimplemented when using different sets of optimizer and schedulers
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            cur_grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            backward_dict.update({"grad_norm": cur_grad_norm})
        self.optimizer.step()
        self.scheduler.step()
        return backward_dict

    def train_step(self, epoch):
        self.model.train()
        loader = self.data_loaders["train"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process), desc=f"[Epoch {epoch + 1}/{self.epochs}]")
        for i, data_dict in enumerate(loader):
            with self.accelerator.accumulate(self.model):
                data_dict = self.forward(data_dict)
                loss, losses = self.loss(data_dict)
                backward_dict = self.backward(loss)
                # record
                self.global_step += 1
                log_dict = {'step': self.global_step}
                log_dict.update(losses)
                log_dict.update(backward_dict)
                self.log(log_dict, mode="train")
                pbar.update(1)

    @torch.no_grad()
    def eval_step(self, epoch):
        self.model.eval()
        target_metrics = []
        loaders = self.data_loaders["val"]
        evaluators = self.evaluator
        for loader, evaluator in zip(loaders, evaluators):
            st = time()
            is_best, results = self.eval(loader, evaluator)
            end = time()
            results['time'] = (end - st) / 60
            self.log(results, mode="val-" + loader.dataset.dataset.__class__.__name__) 
            evaluator.reset()
            target_metrics.append(results['target_metric'])
        target_metric = np.sum(target_metrics)
        if target_metric > self.exp_tracker.best_result:
            is_best = True
            self.exp_tracker.best_result = target_metric
        else:
            is_best = False
        return is_best

    @torch.no_grad()
    def test_step(self):
        self.model.eval()
        loaders = self.data_loaders["test"]
        evaluators = self.evaluator
        for loader, evaluator in zip(loaders, evaluators):
            is_best, results = self.eval(loader, evaluator)
            self.log(results, mode= "test-" + loader.dataset.dataset.__class__.__name__)
            evaluator.reset()
        return results
    
    @torch.no_grad()
    def eval(self, loader, evaluator):
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process), desc=loader.dataset.dataset.__class__.__name__)
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            evaluator.update(data_dict)
            pbar.update(1)
        return evaluator.record()

    def run(self):
        if self.mode == "train":
            start_epoch = self.exp_tracker.epoch
            self.global_step = start_epoch * len(self.data_loaders["train"])
            for epoch in range(start_epoch, self.epochs):
                self.exp_tracker.step()
                st = time()
                self.train_step(epoch)
                epoch_time = (time() - st) / 60
                self.log({"epoch_time": epoch_time, "epoch": epoch+1, "remaining_time": epoch_time * (self.epochs - epoch) / 60}, mode="train")

                if self.epochs_per_eval and (epoch + 1) % self.epochs_per_eval == 0:
                    is_best = self.eval_step(epoch)
                    self.accelerator.print(f"[Epoch {epoch + 1}/{self.epochs}] finished eval, is_best: {is_best}")
                else:
                    is_best = False

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    if is_best:
                        self.save("best.pth")
                    if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                        self.save(f"ckpt_epoch.pth")
                    self.save("latest.pth")

        self.test_step()
        if self.mode == "train":
            self.accelerator.end_training()
