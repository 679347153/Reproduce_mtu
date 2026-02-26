import torch
import torch.nn as nn

from modules.build import HEADS_REGISTRY
from modules.utils import get_mlp_head

@HEADS_REGISTRY.register()
class BoxHead(nn.Module):
    def __init__(self, cfg, input_size=768, hidden_size=384, output_size=6, dropout=0.3):
        super().__init__()
        self.box_head = get_mlp_head(
            input_size, hidden_size,
            output_size, dropout=dropout
        )

    def forward(self, obj_embeds, **kwargs):
        box_prediction = self.box_head(obj_embeds)
        box_size_prediction = torch.exp(box_prediction[:, :, 3:])
        box_prediction = torch.cat([box_prediction[:, :, :3], box_size_prediction], dim=-1)
        return box_prediction
