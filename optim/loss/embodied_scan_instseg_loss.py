import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from optim.criterion.matcher import EmbodiedMatcher, EmbodiedRecurrentMatcher
from optim.criterion.mixed_criterion import MixedCriterion
from optim.loss.loss import LOSS_REGISTRY

from modules.third_party.mask3d.matcher import HungarianMatcher

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
) 

def iou_loss(pred_boxes, target_boxes):
    """
    Forward pass for AxisAlignedIoULoss.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes of shape (N, 6), where each box is 
                                represented as (x, y, z, dx, dy, dz).
        target_boxes (Tensor): Ground truth bounding boxes of shape (N, 6), where each box 
                                is represented as (x, y, z, dx, dy, dz).

    Returns:
        Tensor: Computed IoU loss.
    """
    # Ensure inputs have the correct shape
    assert pred_boxes.shape == target_boxes.shape, "Shape mismatch between predictions and targets"
    assert pred_boxes.shape[-1] == 6, "Boxes must have 6 dimensions (x, y, z, dx, dy, dz)"

    # Extract box centers and sizes
    pred_center = pred_boxes[:, :3]
    pred_size = pred_boxes[:, 3:]
    target_center = target_boxes[:, :3]
    target_size = target_boxes[:, 3:]

    # Compute the min and max corners of the boxes
    pred_min = pred_center - pred_size / 2
    pred_max = pred_center + pred_size / 2
    target_min = target_center - target_size / 2
    target_max = target_center + target_size / 2

    # Compute the intersection box
    inter_min = torch.max(pred_min, target_min)
    inter_max = torch.min(pred_max, target_max)
    inter_size = torch.clamp(inter_max - inter_min, min=0)  # Clamp to avoid negative values

    # Compute the volume of intersection and union
    inter_volume = inter_size[:, 0] * inter_size[:, 1] * inter_size[:, 2]
    pred_volume = pred_size[:, 0] * pred_size[:, 1] * pred_size[:, 2]
    target_volume = target_size[:, 0] * target_size[:, 1] * target_size[:, 2]
    union_volume = pred_volume + target_volume - inter_volume

    # Compute IoU
    iou = inter_volume / torch.clamp(union_volume, min=1e-6)  # Avoid division by zero

    # Compute IoU loss
    iou_loss = 1.0 - iou

    # Apply reduction
    return iou_loss.mean() 

def merge_loss(merge_feat, prev_indices, new_indices, scale=2.30, bias=-10.0):
    if len(prev_indices[0]) == 0 or len(new_indices[0]) == 0:
        return torch.tensor(0.0, device=merge_feat.device)
    # feat sim
    prev_matched_queries = merge_feat[prev_indices[0]]
    new_matched_queries = merge_feat[new_indices[0]]
    prev_new_feat_simlarity = torch.matmul(prev_matched_queries, new_matched_queries.T) # P, N
    prev_new_feat_simlarity = prev_new_feat_simlarity * scale + bias
    # build label Q,Q
    prev_new_feat_label = torch.zeros(prev_new_feat_simlarity.shape, device=prev_new_feat_simlarity.device)
    for i, p_ind in enumerate(prev_indices[0]):
        for j, n_ind in enumerate(new_indices[0]):
            if prev_indices[1][i] == new_indices[1][j]:
                prev_new_feat_label[i, j] = 1.0
    # compute sigmoid cross entropy loss
    loss = F.binary_cross_entropy_with_logits(prev_new_feat_simlarity, prev_new_feat_label, reduction='none') 
    loss = loss.sum(dim=0).mean()
    return loss

@LOSS_REGISTRY.register()
class EmbodiedScanInstSegLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        loss_cfg = cfg.model.get(self.__class__.__name__)

        # training objective
        self.mask_type = "segment_mask"
        # matcher
        self.matcher = EmbodiedMatcher(**loss_cfg.matcher)
        # loss weight
        weight_dict = {"loss_ce": loss_cfg.cost_class,
                    "loss_mask": loss_cfg.cost_mask,
                    "loss_dice": loss_cfg.cost_dice,
                    "loss_score": loss_cfg.cost_score,
                    "loss_box": loss_cfg.cost_box,
                    "loss_open_vocab": loss_cfg.cost_open_vocab}
        aux_weight_dict = {}
        for i in range(len(cfg.model.voxel_encoder.args.hlevels) * cfg.model.unified_encoder.args.num_blocks):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.weight_dict = weight_dict
        # score weight
        self.score_weight = loss_cfg.get('score_weight', [0.1, 1.0])
    
    def get_loss_one_layer(self, outputs, targets, layer_num=-1):
        # match
        indices = self.matcher(outputs, targets)
        # prepase
        losses = {}
        suffix = f'_{layer_num}' if layer_num != -1 else ''
        loss_ce = []
        loss_mask = []
        loss_dice = []
        loss_score = []
        loss_box = []
        loss_open_vocab = []
        losses = {}
        
        # compute loss
        for bid in range(len(outputs)):
            if indices[bid][0].shape[0] == 0:
                continue
            # classification loss
            loss_ce.append(F.cross_entropy(outputs[bid]['pred_classes'], targets[bid]['query_labels']))
            # mask loss
            loss_mask.append(sigmoid_ce_loss_jit(outputs[bid]['pred_masks'][:, indices[bid][0]].T, targets[bid]['masks'][indices[bid][1]].float(), targets[bid]['masks'][indices[bid][1]].shape[0]))
            # dice loss
            loss_dice.append(dice_loss_jit(outputs[bid]['pred_masks'][:, indices[bid][0]].T, targets[bid]['masks'][indices[bid][1]].float(), targets[bid]['masks'][indices[bid][1]].shape[0]))
            # score loss
            score_target = outputs[bid]['pred_scores'].new_zeros(outputs[bid]['pred_scores'].shape[0])
            score_target[indices[bid][0]] = 1.0
            loss_score.append(F.cross_entropy(outputs[bid]['pred_scores'], score_target.long(), weight=score_target.new_tensor(self.score_weight)))
            # box loss
            loss_box.append(iou_loss(outputs[bid]['pred_boxes'][indices[bid][0]], targets[bid]['boxes'][indices[bid][1]])) 
            # open vocab loss
            if layer_num == -1:
                loss_open_vocab.append((1 - torch.nn.CosineSimilarity()
                    (outputs[bid]['pred_embeds'][indices[bid][0]], targets[bid]['instance_text_embeds'][indices[bid][1]])).mean())
        # return dict
        losses['loss_ce' + suffix] = sum(loss_ce) / len(loss_ce) if len(loss_ce) > 0 else 0
        losses['loss_mask' + suffix] = sum(loss_mask) / len(loss_mask) if len(loss_mask) > 0 else 0 
        losses['loss_dice' + suffix] = sum(loss_dice) / len(loss_dice) if len(loss_dice) > 0 else 0
        losses['loss_score' + suffix] = sum(loss_score) / len(loss_score) if len(loss_score) > 0 else 0
        losses['loss_box' + suffix] = sum(loss_box) / len(loss_box) if len(loss_box) > 0 else 0
        if layer_num == -1:
            losses['loss_open_vocab'] = sum(loss_open_vocab) / len(loss_open_vocab) if len(loss_open_vocab) > 0 else 0
        return losses, indices
        
    def forward(self, data_dict):
        # meta data
        bs = data_dict['predictions_mask'][-1].shape[0]
        # load prediction
        predictions_class = data_dict['predictions_class'].copy()
        predictions_mask = data_dict['predictions_mask'].copy()
        predictions_box = data_dict['predictions_box'].copy()
        predictions_score = data_dict['predictions_score'].copy()
        predictions_open_vocab = data_dict['openvocab_query_feat'].clone()
        query_pad_mask = data_dict['query_pad_masks']
        seg_pad_masks = data_dict['seg_pad_masks']
        for l in range(len(predictions_mask)):
            predictions_mask[l] = [predictions_mask[l][bid][:, query_pad_mask[bid].bool()][seg_pad_masks[bid].bool(), :] for bid in range(bs)]
            predictions_class[l] = [predictions_class[l][bid][query_pad_mask[bid].bool(), :] for bid in range(bs)]
            predictions_box[l] = [predictions_box[l][bid][query_pad_mask[bid].bool(), :] for bid in range(bs)]
            predictions_score[l] = [predictions_score[l][bid][query_pad_mask[bid].bool()] for bid in range(bs)]
        predictions_open_vocab = [predictions_open_vocab[bid][query_pad_mask[bid].bool()] for bid in range(bs)]
        # load target
        if self.mask_type == "segment_mask":
            segment_labels = data_dict['segment_labels']
            segment_masks = data_dict['segment_masks']
            instance_boxes = data_dict['instance_boxes']
            instance_labels = data_dict['instance_labels']
            instance_text_embeds = data_dict['instance_text_embeds']
            query_selection_ids = data_dict['query_selection_ids']
            instance_scores = [torch.ones(segment_masks[bid].shape[0], dtype=torch.long, device=segment_masks[bid].device) for bid in range(len(segment_masks))]
            # build target
            targets = [{'masks': segment_masks[bid], 'labels': instance_labels[bid], 'scores': instance_scores[bid], 'boxes': instance_boxes[bid], 'instance_ids': data_dict['instance_ids_ori'][bid], 'instance_text_embeds': instance_text_embeds[bid],
                        'query_gt_mask': segment_masks[bid].T[query_selection_ids[bid], :], 'query_labels': segment_labels[bid][query_selection_ids[bid]]} for bid in range(bs)] 
        else:
            raise NotImplementedError
        # compute loss for last layer
        losses = {}
        inputs = [{'pred_masks': predictions_mask[-1][bid], 'pred_classes': predictions_class[-1][bid], 'pred_boxes': predictions_box[-1][bid], 'pred_scores': predictions_score[-1][bid], 'pred_embeds': predictions_open_vocab[bid]} for bid in range(bs)]
        loss, indices = self.get_loss_one_layer(inputs, targets, -1)
        data_dict['indices'] = indices
        losses.update(loss)
        # compute loss for all layer
        for l in range(len(predictions_mask) - 1):
            inputs = [{'pred_masks': predictions_mask[l][bid], 'pred_classes': predictions_class[l][bid], 'pred_boxes': predictions_box[l][bid], 'pred_scores': predictions_score[l][bid]} for bid in range(bs)]
            loss, _ = self.get_loss_one_layer(inputs, targets, l)
            losses.update(loss)
        # multiply weight
        for k in list(losses.keys()):
            losses[k] *= self.weight_dict[k]
        return [sum(losses.values()), losses]

