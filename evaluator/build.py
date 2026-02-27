import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import open_dict
from fvcore.common.registry import Registry

from common.misc import gather_dict

EVALUATOR_REGISTRY = Registry("EVALUATOR")


class BaseEvaluator():
    def __init__(self, cfg, accelerator):
        self.accelerator = accelerator
        self.best_result = -np.inf
        self.save = cfg.eval.save
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        self.eval_results = []
        self.eval_dict = {}

    def batch_metrics(self, data_dict, include_count=False):
        raise NotImplementedError("Per batch metrics calculation is required for evaluation")

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict, include_count=True)
        for key in metrics.keys():
            if key not in self.eval_dict:
                self.eval_dict[key] = []
            self.eval_dict[key].append(metrics[key])

    @staticmethod
    def _cpu_scalar(x) -> torch.Tensor:
        """Return a 0-dim CPU tensor for safe cross-rank aggregation."""
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.numel() != 1:
                # Defensive: if a metric accidentally returns a vector, reduce it.
                x = x.mean()
            return x.to(device="cpu")
        # Python number
        return torch.tensor(x, device="cpu")

    def record(self):
        self.eval_dict = gather_dict(self.accelerator, self.eval_dict)

        for k, metrics in self.eval_dict.items():
            if not isinstance(metrics, list):
                continue
            # metrics is a list of (value, count)
            values_cpu = []
            counts_cpu = []
            for x in metrics:
                if not (isinstance(x, (tuple, list)) and len(x) >= 2):
                    continue
                values_cpu.append(self._cpu_scalar(x[0]))
                counts_cpu.append(self._cpu_scalar(x[1]))

            if len(values_cpu) == 0:
                continue

            total_value = torch.stack(values_cpu).sum()
            total_count = torch.stack(counts_cpu).sum()

            denom = max(float(total_count.item()), 1.0)
            self.eval_dict[k] = float(total_value.item()) / denom

        if self.save and self.accelerator.is_main_process:
            with (self.save_dir / "results.json").open("w") as f:
                json.dump(self.eval_results, f, indent=4)

        self.eval_dict['target_metric'] = self.eval_dict[self.target_metric]
        if self.eval_dict["target_metric"] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict["target_metric"]
        else:
            is_best = False
        self.eval_dict['best_result'] = self.best_result
        return is_best, self.eval_dict


def get_eval(name, cfg, accelerator, **kwargs):
    """Get an evaluator or a list of evaluators."""
    if isinstance(name, str):
        eval = EVALUATOR_REGISTRY.get(name)(cfg, accelerator, **kwargs)
    else:
        eval = [EVALUATOR_REGISTRY.get(i)(cfg, accelerator, **kwargs) for i in name]
    return eval

def build_eval(cfg, accelerator, **kwargs):
    if cfg.eval.get("train", None) is not None:
        train_eval = get_eval(cfg.eval.train.name, cfg, accelerator, **kwargs)
        val_eval = get_eval(cfg.eval.val.name, cfg, accelerator, **kwargs)
        return {"train": train_eval, "val": val_eval}
    elif cfg.eval.get("name", None) is not None:
        return get_eval(cfg.eval.name, cfg, accelerator, **kwargs)
    else:
        with open_dict(cfg):
            cfg.eval.name = [cfg.data.get(dataset).evaluator for dataset in cfg.data.val]
        return get_eval(cfg.eval.name, cfg, accelerator, **kwargs)