import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling

from modules.build import HEADS_REGISTRY
from modules.utils import get_mlp_head, layer_repeat


@HEADS_REGISTRY.register()
class MaskHead(nn.Module):
    def __init__(self, cfg, hidden_dim, num_targets, filter_out_classes=None):
        super().__init__()
        # task head
        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.class_embed_head = nn.Linear(hidden_dim, num_targets)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.filter_out_classes = filter_out_classes

    def forward(self, query_feat=None, mask_features=None, mask_segments=None, num_pooling_steps=None, ret_attn_mask=True, point2segment=None, gt_attn_mask=None):
            query_feat = self.decoder_norm(query_feat)
            mask_embed = self.mask_embed_head(query_feat)
            outputs_class = self.class_embed_head(query_feat)
            for filter_out_id in self.filter_out_classes:
                outputs_class[..., filter_out_id] = float("-inf")

            output_masks = []
            
            assert point2segment is not None
            output_segments = []
            for i in range(len(mask_segments)):
                output_segments.append(mask_segments[i] @ mask_embed[i].T)
                if gt_attn_mask is not None:
                    output_masks.append(nn.functional.pad(gt_attn_mask[i].transpose(0, 1).float() * 10 - 5, (0, query_feat.shape[1] - gt_attn_mask[i].shape[0]), mode='constant', value=-5)[point2segment[i]])
                else:
                    output_masks.append(output_segments[-1][point2segment[i]])

            output_masks = torch.cat(output_masks)
            outputs_mask = ME.SparseTensor(features=output_masks,
                                        coordinate_manager=mask_features.coordinate_manager,
                                        coordinate_map_key=mask_features.coordinate_map_key)

            if ret_attn_mask:
                attn_mask = outputs_mask
                for _ in range(num_pooling_steps):
                    attn_mask = self.pooling(attn_mask.float())

                attn_mask = ME.SparseTensor(features=(attn_mask.F.detach().sigmoid() < 0.5),
                                            coordinate_manager=attn_mask.coordinate_manager,
                                            coordinate_map_key=attn_mask.coordinate_map_key)

                return outputs_class, output_segments, attn_mask

            return outputs_class, output_segments

@HEADS_REGISTRY.register()
class MaskHeadSegLevel(nn.Module):
    def __init__(self, cfg, hidden_size, num_targets, memories_for_match=['voxel'], filter_out_classes=None, dropout=0.1):
        super().__init__()

        # cls head
        self.cls_head = get_mlp_head(hidden_size, hidden_size, num_targets, dropout=dropout)
        self.query_activation_head = get_mlp_head(hidden_size, hidden_size, 2, dropout=dropout) # 0 no object, 1 has object
        self.filter_out_classes = filter_out_classes

        # mask head
        memories_for_match = [mem for mem in memories_for_match if mem in ['voxel', 'mv', 'pc']]
        mask_pred_layer = MaskPredictionLayer(hidden_size)
        self.mask_pred_list = layer_repeat(mask_pred_layer, len(memories_for_match))

    def forward(self, query, seg_fts_for_match, seg_masks, offline_attn_masks=None, skip_prediction=False):
        if skip_prediction:
            return None, None, offline_attn_masks
        cls_logits = self.cls_head(query)
        cls_logits[..., self.filter_out_classes] = float("-inf")
        query_activation_logits = self.query_activation_head(query)
        
        mask_logits_list = []
        pad_mask_list = []
        for seg_fts, mask_pred_layer in zip(seg_fts_for_match, self.mask_pred_list):
            feat, mask, pos = seg_fts
            mask_logits = mask_pred_layer(query, feat)
            mask_logits_list.append(mask_logits * mask[..., None].logical_not())
            pad_mask_list.append(mask[..., None].logical_not())
        mask_logits = sum(mask_logits_list) / (sum(pad_mask_list) + 1e-8)
        mask_logits[seg_masks] = -1e6
            
        if offline_attn_masks is not None:
            attn_mask = offline_attn_masks
        else:
            attn_mask = mask_logits.sigmoid().permute(0, 2, 1).detach() < 0.5
        return query_activation_logits, cls_logits, mask_logits, attn_mask

@HEADS_REGISTRY.register()
class MaskHeadSegLevelAdaptive(nn.Module):
    def __init__(self, cfg, hidden_size, num_targets, memories_for_match=['voxel'], filter_out_classes=None, dropout=0.1):
        super().__init__()

        # cls head
        self.cls_head = get_mlp_head(hidden_size, hidden_size, num_targets, dropout=dropout)
        self.filter_out_classes = filter_out_classes

        # mask head
        memories_for_match = [mem for mem in memories_for_match if mem in ['voxel', 'mv', 'pc']]
        self.mask_pred_layer = MaskPredictionLayer(hidden_size)
        adaptive_pool_layer = torch.nn.Linear(hidden_size, hidden_size, False)
        self.adaptive_pool_list = layer_repeat(adaptive_pool_layer, len(memories_for_match))

    def forward(self, query, seg_fts_for_match, seg_masks, offline_attn_masks=None, skip_prediction=False):
        if skip_prediction:
            return None, None, offline_attn_masks
        # cls prediction
        cls_logits = self.cls_head(query)
        cls_logits[..., self.filter_out_classes] = float("-inf")
        # pool segment featueres
        seg_fts_pool = []
        for seg_fts, pool_layer in zip(seg_fts_for_match, self.adaptive_pool_list):
            feat, mask, pos = seg_fts
            seg_fts_pool.append(pool_layer(feat) * mask.unsqueeze(2).logical_not())
        seg_fts_pool = sum(seg_fts_pool)
        # mask head
        mask_logits = self.mask_pred_layer(query, seg_fts_pool)
        mask_logits[seg_masks] = -1e6
            
        if offline_attn_masks is not None:
            attn_mask = offline_attn_masks
        else:
            attn_mask = mask_logits.sigmoid().permute(0, 2, 1).detach() < 0.5
        return cls_logits, mask_logits, attn_mask


@HEADS_REGISTRY.register()
class MaskHeadClip(nn.Module):
    def __init__(self, cfg, hidden_dim, anneal_coef=0.07, filter_out_classes=None, clip_feature_path=None):
        super().__init__()
        # task head
        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.clip_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 768)
        ) # map to clip space
        self.register_buffer('clip_scannet200_feature', torch.load(clip_feature_path).cuda())
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.filter_out_classes = filter_out_classes
        self.anneal_coef = anneal_coef
        

    def forward(self, query_feat=None, mask_features=None, mask_segments=None, num_pooling_steps=None, ret_attn_mask=True, point2segment=None):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        query_feat_clip = nn.functional.normalize(self.clip_embed_head(query_feat), p=2, dim=2)
        text_feat_clip = nn.functional.normalize(self.clip_scannet200_feature, p=2, dim=1)
        outputs_class = query_feat_clip @ text_feat_clip.T / self.anneal_coef
        for filter_out_id in self.filter_out_classes:
            outputs_class[..., filter_out_id] = float("-inf")

        output_masks = []
        
        assert point2segment is not None
        output_segments = []
        for i in range(len(mask_segments)):
            output_segments.append(mask_segments[i] @ mask_embed[i].T)
            output_masks.append(output_segments[-1][point2segment[i]])

        output_masks = torch.cat(output_masks)
        outputs_mask = ME.SparseTensor(features=output_masks,
                                    coordinate_manager=mask_features.coordinate_manager,
                                    coordinate_map_key=mask_features.coordinate_map_key)

        if ret_attn_mask:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())

            attn_mask = ME.SparseTensor(features=(attn_mask.F.detach().sigmoid() < 0.5),
                                        coordinate_manager=attn_mask.coordinate_manager,
                                        coordinate_map_key=attn_mask.coordinate_map_key)

            return outputs_class, output_segments, attn_mask

        return outputs_class, output_segments


class MaskPredictionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size, False)
    
    def forward(self, query, key):
        query = self.q_proj(query)
        key = self.k_proj(key)
        logits = torch.einsum('bld, bmd -> blm', key, query)
        return logits
    
@HEADS_REGISTRY.register()
class MaskHeadSegLevelWithBox(nn.Module):
    def __init__(self, cfg, hidden_size, num_targets, memories_for_match=['voxel'], filter_out_classes=None, dropout=0.1):
        super().__init__()

        # cls head
        self.cls_head = get_mlp_head(hidden_size, hidden_size, num_targets, dropout=dropout)
        self.box_head = get_mlp_head(
            hidden_size + 3, hidden_size,
            6, dropout=0.3
        )

        self.query_activation_head = get_mlp_head(hidden_size, hidden_size, 2, dropout=dropout) # 0 no object, 1 has object
        self.filter_out_classes = filter_out_classes

        # mask head
        memories_for_match = [mem for mem in memories_for_match if mem in ['voxel', 'mv', 'pc']]
        mask_pred_layer = MaskPredictionLayer(hidden_size)
        self.mask_pred_list = layer_repeat(mask_pred_layer, len(memories_for_match))

    def forward(self, query, query_locs, seg_fts_for_match, seg_masks, offline_attn_masks=None, skip_prediction=False):
        if skip_prediction:
            return None, None, offline_attn_masks
        cls_logits = self.cls_head(query)
        cls_logits[..., self.filter_out_classes] = float("-inf")
        query_activation_logits = self.query_activation_head(query)
        box_prediction = self.box_head(torch.cat([query, query_locs], dim=-1))
        box_size_prediction = torch.exp(box_prediction[:, :, 3:])
        box_prediction = torch.cat([box_prediction[:, :, :3] + query_locs[:, :, :3], box_size_prediction], dim=-1)
        
        mask_logits_list = []
        pad_mask_list = []
        for seg_fts, mask_pred_layer in zip(seg_fts_for_match, self.mask_pred_list):
            feat, mask, pos = seg_fts
            mask_logits = mask_pred_layer(query, feat)
            mask_logits_list.append(mask_logits * mask[..., None].logical_not())
            pad_mask_list.append(mask[..., None].logical_not())
        mask_logits = sum(mask_logits_list) / (sum(pad_mask_list) + 1e-8)
        mask_logits[seg_masks] = -1e6
            
        if offline_attn_masks is not None:
            attn_mask = offline_attn_masks
        else:
            attn_mask = mask_logits.sigmoid().permute(0, 2, 1).detach() < 0.5
        return query_activation_logits, cls_logits, mask_logits, box_prediction, attn_mask

@HEADS_REGISTRY.register()
class OpenVocabHead(nn.Module):
    def __init__(self, cfg, hidden_dim=768, out_dim=768):
        super().__init__()
        # task head
        self.clip_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        ) # map to clip space
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query_feat):
        query_feat = self.decoder_norm(query_feat)
        query_feat_clip = self.clip_embed_head(query_feat)
        query_feat_clip = nn.functional.normalize(self.clip_embed_head(query_feat), p=2, dim=2)
        
        return query_feat_clip

def save_scannet200_clip_features(path="./clip_scannet200.pth"):
    from data.datasets.constant import CLASS_LABELS_200
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    inputs = tokenizer([lang for lang in CLASS_LABELS_200] + ['no object'], padding=True, return_tensors="pt")
    #inputs = tokenizer(['can sit', 'people can put book on', 'emit light', 'can watch', 'exit this room'] + ['no object'], padding=True, return_tensors="pt")
    outputs = model(**inputs)
    text_embeds = outputs.text_embeds
    torch.save(text_embeds, path)
        
if __name__ == "__main__":
    save_scannet200_clip_features("../saved_features/clip_open_5.pth")