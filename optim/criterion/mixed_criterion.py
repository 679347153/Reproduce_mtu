
import torch
import torch.nn.functional as F
from torch import nn
from copy import deepcopy


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
)  # type: torch.jit.ScriptModule

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
        
class MixedCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses_last_layer,
        losses_all,
        num_points,
        class_weights,
        ignore_label
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses_laster_layer = losses_last_layer
        self.losses_all = losses_all
        self.ignore_label = ignore_label

        if self.class_weights != -1:
            assert (
                len(self.class_weights) == self.num_classes
            ), "CLASS WEIGHTS DO NOT MATCH"

        # pointwise mask loss parameters
        self.num_points = num_points

    def loss_labels(self, outputs, targets, indices, mask_type):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            ignore_index=self.ignore_label,
        )
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_boxes(self, outputs, targets, indices, mask_type):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_boxes"].float()

        idx = self._get_src_permutation_idx(indices)
        src_boxes = torch.cat([src_logits[bid, J] for bid, (J, _) in enumerate(indices)]) # Bx6
        target_boxes = torch.cat(
            [t["boxes"][J] for t, (_, J) in zip(targets, indices)]
        ) # B, 6
    
        loss_box = iou_loss(src_boxes, target_boxes)
        losses = {"loss_box": loss_box}
        return losses

    def loss_masks(self, outputs, targets, indices, mask_type):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id][mask_type][target_id]

            if self.num_points != -1:
                point_idx = torch.randperm(
                    target_mask.shape[1], device=target_mask.device
                )[: int(self.num_points * target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(
                    target_mask.shape[1], device=target_mask.device
                )

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()
            
            if len(map_id) == 0:
                continue
            else:
                loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
                loss_dices.append(dice_loss_jit(map, target_mask, num_masks))
        # del target_mask
        if len(loss_masks) and len(loss_dices):
            return {
                "loss_mask": torch.mean(torch.stack(loss_masks)),
                "loss_dice": torch.mean(torch.stack(loss_dices)),
            }
        else:
            return {
                "loss_mask": torch.tensor(0.0, device=outputs["pred_masks"].device),
                "loss_dice": torch.tensor(0.0, device=outputs["pred_masks"].device),
            }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, mask_type):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks, 'boxes': self.loss_boxes}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, mask_type)

    def forward(self, predictions_mask, predictions_class, predictions_box, prev_instance_ids, instance_ids, instance_labels, instance_boxes, segment_masks, seg_point_count):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """ 
        mask_type = 'segment_masks' 
        seg_point_count = [seg_point_count[i] for i in range(len(seg_point_count))]
        targets = [{'labels': labels, 'segment_masks': masks, 'instance_ids': ids, 'boxes': boxes, 'seg_point_count': seg_count} for labels, masks, ids, boxes, seg_count in zip(instance_labels, segment_masks, instance_ids, instance_boxes, seg_point_count)]

        # Retrieve the matching between the outputs of the last layer and the targets
        last_layer_output = {'pred_logits': predictions_class[-1], 'pred_masks': predictions_mask[-1], 'pred_boxes': predictions_box, 'prev_instance_ids': prev_instance_ids}
        indices = self.matcher(last_layer_output, targets, mask_type)
        ret_indices = deepcopy(indices)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses_laster_layer:
            losses.update(
                self.get_loss(
                    loss, last_layer_output, targets, indices, mask_type
                )
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        for i, (pred_logits, pred_masks) in enumerate(zip(predictions_class[:-1], predictions_mask[:-1])):
            aux_outputs = {'pred_logits': pred_logits, 'pred_masks': pred_masks, 'prev_instance_ids': prev_instance_ids}
            indices = self.matcher(aux_outputs, targets, mask_type)
            for loss in self.losses_all:
                l_dict = self.get_loss(
                    loss,
                    aux_outputs,
                    targets,
                    indices,
                    mask_type,
                )
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses, ret_indices


