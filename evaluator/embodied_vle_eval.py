from collections import defaultdict
from pathlib import Path
import torch

from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator


@EVALUATOR_REGISTRY.register()
class EmbodiedVLEEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        self.target_metric = 'og_acc'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        self.disable_frontier_length = cfg.eval.get('disable_frontier_length', True)
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        metrics = {}
        og_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
        og3d_logits_grounding = data_dict['og3d_logits'].clone()
        og3d_logits_frontier = data_dict['og3d_logits'].clone()
        og3d_logits_grounding[~data_dict['real_obj_pad_masks']] = float('-inf')
        og3d_logits_frontier[data_dict['real_obj_pad_masks']] = float('-inf')
        og3d_pred_grounding = torch.argmax(og3d_logits_grounding, dim=1)
        og3d_pred_frontier = torch.argmax(og3d_logits_frontier, dim=1)
        total_count = len(og_pred)
        # compute og acc 
        total_correct = 0
        for i in range(total_count):
            if data_dict['tgt_object_id'][i, og_pred[i]] == 1:
                total_correct += 1
        metrics['og_acc'] = total_correct
        # compute grounding and frontier acc
        grounding_count = 0
        grounding_correct= 0
        frontier_count = 0
        frontier_correct = 0
        frontier_length_count = defaultdict(int)
        frontier_length_correct = defaultdict(int)
        for i in range(total_count):
            is_object_decision = data_dict['is_object_decision'][i]
            if is_object_decision:
                grounding_count += 1
                if data_dict['tgt_object_id'][i, og3d_pred_grounding[i]] == 1:
                    grounding_correct += 1
            else:
                frontier_count += 1
                frontier_length = data_dict['obj_pad_masks'][i].sum().item() - data_dict['real_obj_pad_masks'][i].sum().item()
                frontier_length_count[frontier_length] += 1
                if data_dict['tgt_object_id'][i, og3d_pred_frontier[i]] == 1:
                    frontier_correct += 1
                    frontier_length_correct[frontier_length] += 1
        metrics['grounding_acc'] = (grounding_correct, grounding_count)
        metrics['frontier_acc'] = (frontier_correct, frontier_count)
        # disable frontier
        if not self.disable_frontier_length:
            for frontier_length in frontier_length_count.keys():
                metrics[f'frontier_length_{frontier_length}_acc'] = (frontier_length_correct[frontier_length], frontier_length_count[frontier_length])
        # query cls acc
        query_cls_correct = 0
        for i in range(total_count):
            labels = data_dict['query_cls_label'][i]
            logits = data_dict['query_cls_logits'][i]
            mask = labels != -100
            query_cls_correct += (logits[mask].argmax(dim=1) == labels[mask]).float().mean()
        metrics['query_cls_acc'] = query_cls_correct 
        # decision acc
        decision_correct = 0
        for i in range(total_count):
            labels = data_dict['decision_label'][i]
            logits = data_dict['decision_logits'][i]
            mask = labels != -100
            decision_correct += (logits[mask].argmax(dim=1) == labels[mask]).float().mean()
        metrics['decision_acc'] = decision_correct

        for key in metrics:
            if isinstance(metrics[key], tuple):
                # already has count
                continue
            metrics[key] = (metrics[key], total_count)

        if self.save:
            item_ids = data_dict['data_idx']
            for i in range(len(item_ids)):
                self.eval_results.append({
                    "scene_id": item_ids[i],
                    "bbox": data_dict['obj_boxes'][i][og_pred[i]].cpu().numpy().tolist(),
                })

        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)

        return metrics
