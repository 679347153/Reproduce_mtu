import gc
import math
import json
import os
from pathlib import Path

import numpy as np
import scipy
import torch
from torch_scatter import scatter_mean
from torch.nn.functional import softmax
from sklearn.cluster import DBSCAN

from common.embodied_utils.instseg_utils import eval_instseg_flexible
from data.data_utils import LabelConverter, pad_sequence
from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator
from common.metric_utils import IoU, ConfusionMatrix
from common.metric_utils import IoU
from common.eval_det import eval_det
from common.eval_instseg import eval_instseg
from common.misc import gather_dict
from data.datasets.constant import VALID_CLASS_IDS_200, VALID_CLASS_IDS_200_VALIDATION, HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from common.embodied_utils.merge_utils import RepresentationManagerGT, RepresentationManager
from data.datasets.hm3d_label_convert import convert_gpt4
from data.datasets.constant import CLASS_LABELS_200

@EVALUATOR_REGISTRY.register()
class EmbodiedScanInstSegEvalEmpty(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        # config
        self.config = cfg
        # record metrics
        self.eval_dict = {'target_metric': []}
        self.target_metric = None
        self.total_count = 0
        self.best_result = -np.inf
        # save dir
        self.save = cfg.eval.save
        if self.save:
            self.eval_results = []
            self.save_dir = Path(cfg.exp_dir) / "eval_results" / "InstSeg"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        # record
        self.preds = defaultdict(dict)
        # misc
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.filter_out_classes = cfg.eval.filter_out_classes
    
    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        self.total_count += metrics["total_count"]

    def batch_metrics(self, data_dict):
        metrics = {}
        metrics["total_count"] = len(data_dict['predictions_class'][0])
        return metrics

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0

    def record(self):        
        # clean
        gc.collect()
        torch.cuda.empty_cache()
        return True, {"target_metric": 0}

@EVALUATOR_REGISTRY.register()
class EmbodiedScanInstSegEvalGTMerge(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        # config
        self.config = cfg
        # record metrics
        self.eval_dict = {'target_metric': []}
        self.target_metric = 'class_acc'
        self.total_count = 0
        self.best_result = -np.inf
        # save dir
        self.save = cfg.eval.save
        if self.save:
            self.save_dir = Path(cfg.exp_dir) / "eval_results" / "InstSeg"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        # representation manager
        self.representation_manger = RepresentationManagerGT()
        self.cur_scan_id = None
        # record
        self.preds = defaultdict(dict)
        # misc
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.filter_out_classes = cfg.eval.filter_out_classes
        self.dataset_name = cfg.eval.dataset_name
    
    def flush_representation_manager(self):
        scan_id = self.cur_scan_id
        self.preds[scan_id] = {}
        for idx, cur_object in enumerate(self.representation_manger.object_id):
            cur_open_vocab_feat = self.representation_manger.open_vocab_feat[idx]
            cur_class = np.argmax(self.representation_manger.object_class[idx])
            self.preds[scan_id][int(cur_object)] = {'open_vocab_feat': cur_open_vocab_feat.copy(), 'class_id': int(cur_class)}
        self.representation_manger.reset()
        
    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        # check whether dump data from representation manager to preds
        scan_id = data_dict['scan_id'][0]
        sub_frame_id = data_dict['sub_frame_id'][0]
        if self.cur_scan_id is None:
            self.cur_scan_id = scan_id
        elif self.cur_scan_id != scan_id:
            self.flush_representation_manager()
            self.cur_scan_id = scan_id
        # merge
        pred_masks = data_dict['predictions_mask'][-1]
        pred_logits = data_dict['predictions_class'][-1]
        pred_boxes = data_dict['predictions_box'][-1]    
        pred_scores = data_dict['predictions_score'][-1]
        pred_feats = data_dict['query_feat']
        pred_embeds = data_dict['openvocab_query_feat']
        query_pad_masks = data_dict['query_pad_masks']
        voxel_to_full_maps = data_dict['voxel_to_full_maps']
        voxel2segment = data_dict['voxel2segment']
        segment_to_full_maps = data_dict['segment_to_full_maps']
        raw_coordinates = data_dict['raw_coordinates']
        pred_indices = data_dict['indices']
        instance_ids_ori = data_dict['instance_ids_ori']
        assert len(pred_masks) == 1
        for bid in range(len(pred_masks)):
            # get all stuff
            masks = pred_masks[bid].detach().cpu()[voxel2segment[bid].cpu()][:, query_pad_masks[bid].cpu()]
            logits = pred_logits[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 201)
            for cls in self.filter_out_classes:
                logits[:, cls] = -float('inf')
            boxes = pred_boxes[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 6)
            scores = softmax(pred_scores[bid].detach().cpu()[query_pad_masks[bid].cpu()], dim=1)[:, 1] # (q)
            embeds = pred_embeds[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 768)
            feats = pred_feats[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 256)
            indices = pred_indices[bid]
            indices[0] = indices[0].detach().cpu()
            indices[1] = indices[1].detach().cpu()
            ids = instance_ids_ori[bid].detach().cpu()
            # filter out classes
            masks = masks[:, indices[0]]
            logits = logits[indices[0]]
            boxes = boxes[indices[0]]
            scores = scores[indices[0]]
            embeds = embeds[indices[0]]
            feats = feats[indices[0]]
            ids = ids[indices[1]]
            if masks.shape[1] == 0:
                continue
            # get mask scores heatmap
            mask_scores, masks, classes, heatmap = get_mask_and_scores(logits, masks)
            # convert mask to full res
            masks = get_full_res_mask(masks, voxel_to_full_maps[bid].cpu(), segment_to_full_maps[bid].cpu())
            masks = masks.numpy()
            classes = classes.numpy()
            boxes = boxes.numpy()
            mask_scores = mask_scores.numpy()
            scores = scores.numpy()
            feats = feats.numpy()
            embeds = embeds.numpy()
            ids = ids.numpy()
            # merge
            predict_dict_list = [
            {
                'point_cloud': raw_coordinates[bid],
                'pred_masks': masks,
                'pred_classes': classes,
                'pred_scores': scores,
                'pred_mask_scores': mask_scores,
                'pred_boxes': boxes,
                'pred_feats': feats,
                'open_vocab_feats': embeds,
                'pred_ids': ids,
            }]
            self.representation_manger.merge(predict_dict_list)
            
        # update representation
        self.total_count += metrics["total_count"]

    def batch_metrics(self, data_dict):
        metrics = {}
        metrics["total_count"] = len(data_dict['predictions_class'][0])
        assert metrics["total_count"] == 1
        return metrics

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0
        self.preds = defaultdict(dict)
        self.representation_manger.reset()

    def record(self):        
        self.flush_representation_manager()
        # gather for metrics
        data = {'preds': list(self.preds.items())}
        data = gather_dict(self.accelerator, data)
        self.preds = dict(data['preds'])
        # judge result
        if self.dataset_name == 'HM3D':
            eval_results = eval_semantic_hm3d(self.preds, self.config)
        elif self.dataset_name == 'ScanNet':
            eval_results = eval_semantic_scannet(self.preds, self.config)
        else:
            raise NotImplementedError
        if eval_results[self.target_metric] > self.best_result:
            is_best = True
            self.best_result = eval_results[self.target_metric]
        else:
            is_best = False
        eval_results['target_metric'] = eval_results[self.target_metric]
        eval_results['best_result'] = self.best_result
        # clean
        gc.collect()
        torch.cuda.empty_cache()
        return is_best, eval_results


@EVALUATOR_REGISTRY.register()
class EmbodiedScanInstSegEvalBoxMerge(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        # config
        self.config = cfg
        # record metrics
        self.eval_dict = {'target_metric': []}
        self.target_metric = 'all_ap'
        self.total_count = 0
        self.best_result = -np.inf
        # save dir
        self.save = cfg.eval.save
        if self.save:
            self.save_dir = Path(cfg.exp_dir) / "eval_results" / "InstSeg"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        # representation manager
        self.representation_manger = RepresentationManager()
        self.cur_scan_id = None
        # record
        self.preds = defaultdict(dict)
        # misc
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.filter_out_classes = cfg.eval.filter_out_classes
        self.dataset_name = cfg.eval.dataset_name
    
    def flush_representation_manager(self):
        scan_id = self.cur_scan_id
        self.preds[scan_id] = {}
        # we only need pred_scores, pred_masks, pred_classes
        pred_point_cloud = self.representation_manger.point_cloud # Nx6
        pred_masks = self.representation_manger.object_mask # NxM
        pred_scores = self.representation_manger.object_score # N
        pred_classes = np.argmax(self.representation_manger.object_class, axis=1) # N
        if self.dataset_name == 'HM3D':
            gt_pcd_data = np.fromfile(os.path.join(self.config.data.embodied_base, 'HM3D', 'points_global', scan_id + '.bin'), dtype=np.float32).reshape(-1, 6)
            points, colors = gt_pcd_data[:, :3], gt_pcd_data[:, 3:]
            points[:, [1, 2]] = points[:, [2, 1]]
        elif self.dataset_name == 'ScanNet':
            gt_pcd_data = np.fromfile(os.path.join(self.config.data.embodied_base, 'ScanNet', 'points_global', scan_id + '.bin'), dtype=np.float32).reshape(-1, 6)
            points, colors = gt_pcd_data[:, :3], gt_pcd_data[:, 3:]
        kdtree = scipy.spatial.cKDTree(pred_point_cloud[:, :3])
        distance, indices = kdtree.query(points[:, :3], k=1)
        pred_masks = pred_masks[indices]
        # polish mask by segment from reconstructed point cloud like ESAM
        if self.dataset_name == 'ScanNet':
            segment_id_path = os.path.join(self.config.data.embodied_base, 'ScanNet', 'segment_id_global', scan_id + '.npy') 
            pred_masks = torch.from_numpy(pred_masks).float().T # (M, N)
            segment_id = np.load(segment_id_path)
            assert segment_id.max() == (np.unique(segment_id).shape[0] - 1)
            segment_id = torch.from_numpy(segment_id).long() # (N)
            pred_masks = scatter_mean(pred_masks, segment_id, dim=1) # (M, S)
            pred_masks = (pred_masks > 0.5)[:, segment_id] # (M, N)
            pred_masks = pred_masks.T.numpy()            
        # fill preds
        self.preds[scan_id] = {'pred_scores': pred_scores, 'pred_masks': pred_masks, 'pred_classes': pred_classes}        
        # reset
        self.representation_manger.reset()
        
    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        # check whether dump data from representation manager to preds
        scan_id = data_dict['scan_id'][0]
        sub_frame_id = data_dict['sub_frame_id'][0]
        if self.cur_scan_id is None:
            self.cur_scan_id = scan_id
        elif self.cur_scan_id != scan_id:
            self.flush_representation_manager()
            self.cur_scan_id = scan_id
        # merge
        pred_masks = data_dict['predictions_mask'][-1]
        pred_logits = data_dict['predictions_class'][-1]
        pred_boxes = data_dict['predictions_box'][-1]    
        pred_scores = data_dict['predictions_score'][-1]
        pred_feats = data_dict['query_feat']
        pred_embeds = data_dict['openvocab_query_feat']
        query_pad_masks = data_dict['query_pad_masks']
        voxel_to_full_maps = data_dict['voxel_to_full_maps']
        voxel2segment = data_dict['voxel2segment']
        segment_to_full_maps = data_dict['segment_to_full_maps']
        raw_coordinates = data_dict['raw_coordinates']
        assert len(pred_masks) == 1
        for bid in range(len(pred_masks)):
            # get all stuff
            masks = pred_masks[bid].detach().cpu()[voxel2segment[bid].cpu()][:, query_pad_masks[bid].cpu()]
            logits = pred_logits[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 201)
            boxes = pred_boxes[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 6)
            scores = softmax(pred_scores[bid].detach().cpu()[query_pad_masks[bid].cpu()], dim=1)[:, 1] # (q)
            embeds = pred_embeds[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 768)
            feats = pred_feats[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 256)
            # filter out classes
            valid_query_mask = ~torch.isin(torch.argmax(logits, dim=-1), torch.tensor(self.filter_out_classes))
            # filter out classes
            masks = masks[:, valid_query_mask]
            logits = logits[valid_query_mask]
            boxes = boxes[valid_query_mask]
            scores = scores[valid_query_mask]
            embeds = embeds[valid_query_mask]
            feats = feats[valid_query_mask]
            if masks.shape[1] == 0:
                continue
            # remove 200 classes
            logits = logits[:, :200]
            # get mask scores heatmap
            mask_scores, masks, classes, heatmap = get_mask_and_scores(logits, masks)
            # convert mask to full res
            masks = get_full_res_mask(masks, voxel_to_full_maps[bid].cpu(), segment_to_full_maps[bid].cpu())
            masks = masks.numpy()
            classes = classes.numpy()
            boxes = boxes.numpy()
            mask_scores = mask_scores.numpy()
            scores = scores.numpy()
            feats = feats.numpy()
            embeds = embeds.numpy()
            # merge
            predict_dict_list = [
            {
                'point_cloud': raw_coordinates[bid],
                'pred_masks': masks,
                'pred_classes': classes,
                'pred_scores': scores,
                'pred_mask_scores': mask_scores,
                'pred_boxes': boxes,
                'pred_feats': feats,
                'open_vocab_feats': embeds,
            }]
            self.representation_manger.merge(predict_dict_list)
            
        # update representation
        self.total_count += metrics["total_count"]

    def batch_metrics(self, data_dict):
        metrics = {}
        metrics["total_count"] = len(data_dict['predictions_class'][0])
        assert metrics["total_count"] == 1
        return metrics

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0
        self.preds = defaultdict(dict)
        self.representation_manger.reset()

    def record(self):        
        self.flush_representation_manager()
        # gather for metrics
        data = {'preds': list(self.preds.items())}
        data = gather_dict(self.accelerator, data)
        self.preds = dict(data['preds'])
        # judge result
        if self.dataset_name == 'HM3D':
            eval_results = eval_mask_hm3d(self.preds, self.config)
        elif self.dataset_name == 'ScanNet':
            eval_results = eval_mask_scannet(self.preds, self.config)
        else:
            raise NotImplementedError
        if eval_results[self.target_metric] > self.best_result:
            is_best = True
            self.best_result = eval_results[self.target_metric]
        else:
            is_best = False
        eval_results['target_metric'] = eval_results[self.target_metric]
        eval_results['best_result'] = self.best_result
        # clean
        gc.collect()
        torch.cuda.empty_cache()
        return is_best, eval_results

@EVALUATOR_REGISTRY.register()
class EmbodiedScanInstSegEvalBoxMergeOpenVocab(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        # config
        self.config = cfg
        # record metrics
        self.eval_dict = {'target_metric': []}
        self.target_metric = 'all_ap'
        self.total_count = 0
        self.best_result = -np.inf
        # save dir
        self.save = cfg.eval.save
        if self.save:
            self.save_dir = Path(cfg.exp_dir) / "eval_results" / "InstSeg"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        # representation manager
        self.representation_manger = RepresentationManager()
        self.cur_scan_id = None
        # record
        self.preds = defaultdict(dict)
        # misc
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.filter_out_classes = cfg.eval.filter_out_classes
        self.dataset_name = cfg.eval.dataset_name
    
    def flush_representation_manager(self):
        scan_id = self.cur_scan_id
        self.preds[scan_id] = {}
        # we only need pred_scores, pred_masks, pred_classes
        pred_point_cloud = self.representation_manger.point_cloud # Nx6
        pred_masks = self.representation_manger.object_mask # NxM
        pred_scores = self.representation_manger.object_score # M
        pred_classes = np.argmax(self.representation_manger.object_class, axis=1) # M
        pred_openvocab = self.representation_manger.open_vocab_feat # MxD
        if self.dataset_name == 'HM3D':
            gt_pcd_data = np.fromfile(os.path.join(self.config.data.embodied_base, 'HM3D', 'points_global', scan_id + '.bin'), dtype=np.float32).reshape(-1, 6)
            points, colors = gt_pcd_data[:, :3], gt_pcd_data[:, 3:]
            points[:, [1, 2]] = points[:, [2, 1]]
        elif self.dataset_name == 'ScanNet':
            gt_pcd_data = np.fromfile(os.path.join(self.config.data.embodied_base, 'ScanNet', 'points_global', scan_id + '.bin'), dtype=np.float32).reshape(-1, 6)
            points, colors = gt_pcd_data[:, :3], gt_pcd_data[:, 3:]
        kdtree = scipy.spatial.cKDTree(pred_point_cloud[:, :3])
        distance, indices = kdtree.query(points[:, :3], k=1)
        pred_masks = pred_masks[indices]
        # polish mask by segment from reconstructed point cloud like ESAM
        if self.dataset_name == 'ScanNet':
            segment_id_path = os.path.join(self.config.data.embodied_base, 'ScanNet', 'segment_id_global', scan_id + '.npy') 
            pred_masks = torch.from_numpy(pred_masks).float().T # (M, N)
            segment_id = np.load(segment_id_path)
            assert segment_id.max() == (np.unique(segment_id).shape[0] - 1)
            segment_id = torch.from_numpy(segment_id).long() # (N)
            pred_masks = scatter_mean(pred_masks, segment_id, dim=1) # (M, S)
            pred_masks = (pred_masks > 0.5)[:, segment_id] # (M, N)
            pred_masks = pred_masks.T.numpy()            
        # fill preds
        self.preds[scan_id] = {'pred_scores': pred_scores, 'pred_masks': pred_masks, 'pred_openvocab': pred_openvocab}        
        # reset
        self.representation_manger.reset()
        
    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        # check whether dump data from representation manager to preds
        scan_id = data_dict['scan_id'][0]
        sub_frame_id = data_dict['sub_frame_id'][0]
        if self.cur_scan_id is None:
            self.cur_scan_id = scan_id
        elif self.cur_scan_id != scan_id:
            self.flush_representation_manager()
            self.cur_scan_id = scan_id
        # merge
        pred_masks = data_dict['predictions_mask'][-1]
        pred_logits = data_dict['predictions_class'][-1]
        pred_boxes = data_dict['predictions_box'][-1]    
        pred_scores = data_dict['predictions_score'][-1]
        pred_feats = data_dict['query_feat']
        pred_embeds = data_dict['openvocab_query_feat']
        query_pad_masks = data_dict['query_pad_masks']
        voxel_to_full_maps = data_dict['voxel_to_full_maps']
        voxel2segment = data_dict['voxel2segment']
        segment_to_full_maps = data_dict['segment_to_full_maps']
        raw_coordinates = data_dict['raw_coordinates']
        assert len(pred_masks) == 1
        for bid in range(len(pred_masks)):
            # get all stuff
            masks = pred_masks[bid].detach().cpu()[voxel2segment[bid].cpu()][:, query_pad_masks[bid].cpu()]
            logits = pred_logits[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 201)
            boxes = pred_boxes[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 6)
            scores = softmax(pred_scores[bid].detach().cpu()[query_pad_masks[bid].cpu()], dim=1)[:, 1] # (q)
            embeds = pred_embeds[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 768)
            feats = pred_feats[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 256)
            # filter out classes
            valid_query_mask = ~torch.isin(torch.argmax(logits, dim=-1), torch.tensor(self.filter_out_classes))
            # filter out classes
            masks = masks[:, valid_query_mask]
            logits = logits[valid_query_mask]
            boxes = boxes[valid_query_mask]
            scores = scores[valid_query_mask]
            embeds = embeds[valid_query_mask]
            feats = feats[valid_query_mask]
            if masks.shape[1] == 0:
                continue
            # remove 200 classes
            logits = logits[:, :200]
            # get mask scores heatmap
            mask_scores, masks, classes, heatmap = get_mask_and_scores(logits, masks)
            # convert mask to full res
            masks = get_full_res_mask(masks, voxel_to_full_maps[bid].cpu(), segment_to_full_maps[bid].cpu())
            masks = masks.numpy()
            classes = classes.numpy()
            boxes = boxes.numpy()
            mask_scores = mask_scores.numpy()
            scores = scores.numpy()
            feats = feats.numpy()
            embeds = embeds.numpy()
            # merge
            predict_dict_list = [
            {
                'point_cloud': raw_coordinates[bid],
                'pred_masks': masks,
                'pred_classes': classes,
                'pred_scores': scores,
                'pred_mask_scores': mask_scores,
                'pred_boxes': boxes,
                'pred_feats': feats,
                'open_vocab_feats': embeds,
            }]
            self.representation_manger.merge(predict_dict_list)
            
        # update representation
        self.total_count += metrics["total_count"]

    def batch_metrics(self, data_dict):
        metrics = {}
        metrics["total_count"] = len(data_dict['predictions_class'][0])
        assert metrics["total_count"] == 1
        return metrics

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0
        self.preds = defaultdict(dict)
        self.representation_manger.reset()

    def record(self):        
        self.flush_representation_manager()
        # gather for metrics
        data = {'preds': list(self.preds.items())}
        data = gather_dict(self.accelerator, data)
        self.preds = dict(data['preds'])
        # judge result
        if self.dataset_name == 'HM3D':
            eval_results = eval_mask_open_vocab_hm3d(self.preds, self.config)
        elif self.dataset_name == 'ScanNet':
            eval_results = eval_mask_open_vocab_scannet(self.preds, self.config)
        else:
            raise NotImplementedError
        if eval_results[self.target_metric] > self.best_result:
            is_best = True
            self.best_result = eval_results[self.target_metric]
        else:
            is_best = False
        eval_results['target_metric'] = eval_results[self.target_metric]
        eval_results['best_result'] = self.best_result
        # clean
        gc.collect()
        torch.cuda.empty_cache()
        return is_best, eval_results

@EVALUATOR_REGISTRY.register()
class EmbodiedScanInstSegEvalGTMergeSaveFeat(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        # config
        self.config = cfg
        # record metrics
        self.eval_dict = {'target_metric': []}
        self.target_metric = 'class_acc'
        self.total_count = 0
        self.best_result = -np.inf
        # save dir
        self.save = cfg.eval.save
        if self.save:
            self.save_dir = cfg.eval.save_dir
        self.save_frame_interval = cfg.eval.save_frame_interval
        # representation manager
        self.representation_manger = RepresentationManagerGT()
        self.cur_scan_id = None
        # record
        self.preds = defaultdict(lambda: defaultdict(list)) # self.preds[object_id][attribute] = [] list 
        # misc
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.filter_out_classes = cfg.eval.filter_out_classes
        self.dataset_name = cfg.eval.dataset_name
    
    def flush_single_frame(self):
        for idx, cur_object in enumerate(self.representation_manger.object_id):
            if cur_object not in self.preds.keys() or self.representation_manger.object_count[idx] > self.preds[cur_object]['object_count'][-1]:
                self.preds[cur_object]['object_count'].append(self.representation_manger.object_count[idx])
                self.preds[cur_object]['object_id'].append(cur_object)
                # useful for training model
                self.preds[cur_object]['object_score'].append(self.representation_manger.object_score[idx])
                self.preds[cur_object]['object_box'].append(self.representation_manger.object_box[idx])
                self.preds[cur_object]['object_feat'].append(self.representation_manger.object_feat[idx])
                self.preds[cur_object]['object_open_vocab_feat'].append(self.representation_manger.open_vocab_feat[idx])
        self.representation_manger.reset()
    
    def flush_representation_manager(self):
        self.flush_single_frame()
        scan_id = self.cur_scan_id
        save_dict = {}
        for key in map(int, self.preds.keys()):
            save_dict[key] = {}
            save_dict[key]['object_score'] = self.preds[key]['object_score']
            save_dict[key]['object_box'] = self.preds[key]['object_box']
            save_dict[key]['object_feat'] = self.preds[key]['object_feat']
            save_dict[key]['object_open_vocab_feat'] = self.preds[key]['object_open_vocab_feat']
        torch.save(save_dict, os.path.join(self.save_dir, f"{scan_id}.pth"))
        self.representation_manger.reset()
        self.preds = defaultdict(lambda: defaultdict(list))
        
    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        # check whether dump data from representation manager to preds
        scan_id = data_dict['scan_id'][0]
        sub_frame_id = data_dict['sub_frame_id'][0]
        if self.cur_scan_id is None: # no memory
            self.cur_scan_id = scan_id
        elif self.cur_scan_id != scan_id: # new scan
            self.flush_representation_manager()
            self.cur_scan_id = scan_id
        else: # same scan
            if self.save_frame_interval != -1 and int(sub_frame_id) % self.save_frame_interval == 0:
                self.flush_single_frame()
        # merge
        pred_masks = data_dict['predictions_mask'][-1]
        pred_logits = data_dict['predictions_class'][-1]
        pred_boxes = data_dict['predictions_box'][-1]    
        pred_scores = data_dict['predictions_score'][-1]
        pred_feats = data_dict['query_feat']
        pred_embeds = data_dict['openvocab_query_feat']
        query_pad_masks = data_dict['query_pad_masks']
        voxel_to_full_maps = data_dict['voxel_to_full_maps']
        voxel2segment = data_dict['voxel2segment']
        segment_to_full_maps = data_dict['segment_to_full_maps']
        raw_coordinates = data_dict['raw_coordinates']
        pred_indices = data_dict['indices']
        instance_ids_ori = data_dict['instance_ids_ori']
        assert len(pred_masks) == 1
        for bid in range(len(pred_masks)):
            # get all stuff
            masks = pred_masks[bid].detach().cpu()[voxel2segment[bid].cpu()][:, query_pad_masks[bid].cpu()]
            logits = pred_logits[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 201)
            for cls in self.filter_out_classes:
                logits[:, cls] = -float('inf')
            boxes = pred_boxes[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 6)
            scores = softmax(pred_scores[bid].detach().cpu()[query_pad_masks[bid].cpu()], dim=1)[:, 1] # (q)
            embeds = pred_embeds[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 768)
            feats = pred_feats[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 256)
            indices = pred_indices[bid]
            indices[0] = indices[0].detach().cpu()
            indices[1] = indices[1].detach().cpu()
            ids = instance_ids_ori[bid].detach().cpu()
            # filter out classes
            masks = masks[:, indices[0]]
            logits = logits[indices[0]]
            boxes = boxes[indices[0]]
            scores = scores[indices[0]]
            embeds = embeds[indices[0]]
            feats = feats[indices[0]]
            ids = ids[indices[1]]
            if masks.shape[1] == 0:
                continue
            # get mask scores heatmap
            mask_scores, masks, classes, heatmap = get_mask_and_scores(logits, masks)
            # convert mask to full res
            masks = get_full_res_mask(masks, voxel_to_full_maps[bid].cpu(), segment_to_full_maps[bid].cpu())
            masks = masks.numpy()
            classes = classes.numpy()
            boxes = boxes.numpy()
            mask_scores = mask_scores.numpy()
            scores = scores.numpy()
            feats = feats.numpy()
            embeds = embeds.numpy()
            ids = ids.numpy()
            # merge
            predict_dict_list = [
            {
                'point_cloud': raw_coordinates[bid],
                'pred_masks': masks,
                'pred_classes': classes,
                'pred_scores': scores,
                'pred_mask_scores': mask_scores,
                'pred_boxes': boxes,
                'pred_feats': feats,
                'open_vocab_feats': embeds,
                'pred_ids': ids,
            }]
            self.representation_manger.merge(predict_dict_list)
            
        # update representation
        self.total_count += metrics["total_count"]

    def batch_metrics(self, data_dict):
        metrics = {}
        metrics["total_count"] = len(data_dict['predictions_class'][0])
        assert metrics["total_count"] == 1
        return metrics

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0
        self.preds = defaultdict(dict)
        self.representation_manger.reset()

    def record(self):        
        self.flush_representation_manager()
        # clean
        gc.collect()
        torch.cuda.empty_cache()
        return True, {'target_metric': 0, 'best_result': 0}

def get_mask_and_scores(logits, masks):
    labels_per_query = torch.argmax(logits, dim=1)
    
    result_pred_mask = (masks > 0).float()
    heatmap = masks.float().sigmoid()
    
    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
    score = mask_scores_per_image
    classes = labels_per_query
    
    return score, result_pred_mask, classes, heatmap

def get_full_res_mask( mask, voxel_to_full_maps, segment_to_full_maps):
    mask = mask.detach().cpu()[voxel_to_full_maps]  # full res

    segment_to_full_maps = segment_to_full_maps
    mask = scatter_mean(mask, segment_to_full_maps, dim=0)  # full res segments
    mask = (mask > 0.5).float()
    mask = mask.detach().cpu()[segment_to_full_maps]  # full res points

    return mask
    
# evaluators, eval_semantic_hm3d, eval_semantic_scannet, eval_mask_scanneet, eval_mask_hm3d 
# inputs preds {'scan_id': {'object_id': {'open_vocab_feat': open_vocab_feat, 'class_id': class_id}}}
def eval_semantic_hm3d(preds, cfg):
    # load meta data
    base_dir = cfg.data.scene_verse_base
    embodied_base_dir = cfg.data.embodied_base
    # load label converter
    int2cat = json.load(open(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
    cat2int = {w: i for i, w in enumerate(int2cat)}
    label_converter = LabelConverter(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2-labels.combined.tsv"))
    scannet_607_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "ScanNet", "scannet_sem_text_feature.pth"))
    hm3d_sem_category_mapping = np.loadtxt(os.path.join(embodied_base_dir, "HM3D", "hm3dsem_category_mappings.tsv"), dtype=str, delimiter="\t", encoding="utf-8")
    hm3d_raw_to_scannet607 = convert_gpt4
    hm3d_raw_to_cat = {hm3d_sem_category_mapping[i, 0]: hm3d_sem_category_mapping[i, 1] for i in range(1, hm3d_sem_category_mapping.shape[0])}
    hm3d_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "HM3D", "hm3d_sem_text_feature.pth"))
    # load inst label to gts
    gts = {}
    eval_scan_ids = preds.keys()
    for scan_id in eval_scan_ids:
        scan_id_ori = scan_id.split('_')[0]
        inst_to_label = torch.load(os.path.join(embodied_base_dir, 'HM3D','instance_id_to_label', f'{scan_id_ori}_00.pth'))
        inst_to_text_label = {k - 1: hm3d_raw_to_cat[v] if v in hm3d_raw_to_cat.keys() else v for k,v in inst_to_label.items()}
        inst_to_label = {k - 1: hm3d_raw_to_scannet607[v] if v in hm3d_raw_to_scannet607.keys() else 'object' for k, v in inst_to_label.items()}
        gts[scan_id] = {}
        for obj_id in inst_to_label.keys():
            obj_607_name = inst_to_label[obj_id]
            obj_raw_id = label_converter.raw_name_to_scannet_raw_id[obj_607_name]
            obj_scannet_200_label = label_converter.scannet_raw_id_to_scannet200_id[obj_raw_id] if obj_raw_id in label_converter.scannet_raw_id_to_scannet200_id.keys() else -100
            if obj_scannet_200_label in [-100, 0, 2, 35]:
                continue
            obj_sem_feat = hm3d_cat_to_text_embed[inst_to_text_label[obj_id]]
            gts[scan_id][obj_id] = {'class_id': obj_scannet_200_label, 'open_vocab_feat': obj_sem_feat}
    # eval
    class_correct = 0
    class_total = 0
    sem_sim_sum = 0
    correct_sim_sum = 0
    for scan_id in eval_scan_ids:
        for obj_id in preds[scan_id].keys():
            if obj_id not in gts[scan_id].keys():
                continue
            class_total += 1
            cur_sim = torch.cosine_similarity(torch.Tensor(preds[scan_id][obj_id]['open_vocab_feat']), gts[scan_id][obj_id]['open_vocab_feat'], dim=0)
            if preds[scan_id][obj_id]['class_id'] == gts[scan_id][obj_id]['class_id']:
                class_correct += 1
                correct_sim_sum += cur_sim
            sem_sim_sum += cur_sim
    # print result
    class_acc = class_correct / class_total
    sem_sim = sem_sim_sum / class_total
    correct_sem_sim = correct_sim_sum / class_correct
    incorrect_sem_sim = (sem_sim_sum - correct_sim_sum) / (class_total - class_correct)
    print(f"semantic class accuracy: {class_acc}, semantic similarity: {sem_sim}, correct semantic similarity: {correct_sem_sim}, incorrect semantic similarity: {incorrect_sem_sim}")
    return {'class_acc': class_acc, 'sem_sim': sem_sim}

# inputs preds {'scan_id': 'pred_masks': masks, 'pred_classes': classes, 'pred_scores': scores}
def eval_mask_hm3d(preds, cfg): 
    # load meta data
    base_dir = cfg.data.scene_verse_base
    embodied_base_dir = cfg.data.embodied_base
    # load label converter
    int2cat = json.load(open(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
    cat2int = {w: i for i, w in enumerate(int2cat)}
    label_converter = LabelConverter(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2-labels.combined.tsv"))
    scannet_607_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "ScanNet", "scannet_sem_text_feature.pth"))
    hm3d_sem_category_mapping = np.loadtxt(os.path.join(embodied_base_dir, "HM3D", "hm3dsem_category_mappings.tsv"), dtype=str, delimiter="\t", encoding="utf-8")
    hm3d_raw_to_scannet607 = convert_gpt4
    hm3d_raw_to_cat = {hm3d_sem_category_mapping[i, 0]: hm3d_sem_category_mapping[i, 1] for i in range(1, hm3d_sem_category_mapping.shape[0])}
    hm3d_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "HM3D", "hm3d_sem_text_feature.pth")) 
    # build gt_ids
    gt_ids = {}
    valid_scan_ids = list(preds.keys())
    for scan_id in valid_scan_ids:
        scan_id_ori = scan_id.split('_')[0]
        instance_labels = np.fromfile(os.path.join(embodied_base_dir, 'HM3D', 'instance_mask_global',  f'{scan_id}.bin'), dtype=np.int64).reshape(-1)
        inst_to_label = torch.load(os.path.join(embodied_base_dir, 'HM3D','instance_id_to_label', f'{scan_id_ori}_00.pth'))
        gt_mask_list = []
        for inst_id in inst_to_label.keys():
            if inst_to_label[inst_id] in convert_gpt4.keys() and  label_converter.raw_name_to_scannet_raw_id[convert_gpt4[inst_to_label[inst_id]]] in label_converter.scannet_raw_id_to_scannet200_id and convert_gpt4[inst_to_label[inst_id]] not in ['wall', 'floor', 'ceiling']:
                gt_mask_list.append(instance_labels == inst_id)
        preds[scan_id]['pred_classes'] = np.ones(len(preds[scan_id]['pred_classes']))
        cur_ids = np.zeros(preds[scan_id]['pred_masks'].shape[0])
        for i, mask in enumerate(gt_mask_list):
            cur_ids[mask] = 1 * 1000 + i + 1 # class * 1000 + object
        gt_ids[scan_id] = cur_ids
    # eval
    eval_results = eval_instseg_flexible(preds, gt_ids, ['object'])
    print(eval_results)
    return eval_results

# inputs preds {'scan_id': {'object_id': {'open_vocab_feat': open_vocab_feat, 'class_id': class_id}}}
def eval_semantic_scannet(preds, cfg):
    # load meta data
    base_dir = cfg.data.scene_verse_base
    embodied_base_dir = cfg.data.embodied_base
    # load label converter
    int2cat = json.load(open(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
    cat2int = {w: i for i, w in enumerate(int2cat)}
    label_converter = LabelConverter(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2-labels.combined.tsv"))
    scannet_607_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "ScanNet", "scannet_sem_text_feature.pth"))
    hm3d_sem_category_mapping = np.loadtxt(os.path.join(embodied_base_dir, "HM3D", "hm3dsem_category_mappings.tsv"), dtype=str, delimiter="\t", encoding="utf-8")
    hm3d_raw_to_scannet607 = convert_gpt4
    hm3d_raw_to_cat = {hm3d_sem_category_mapping[i, 0]: hm3d_sem_category_mapping[i, 1] for i in range(1, hm3d_sem_category_mapping.shape[0])}
    hm3d_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "HM3D", "hm3d_sem_text_feature.pth"))
    # load inst label to gts
    gts = {}
    eval_scan_ids = preds.keys()
    for scan_id in eval_scan_ids:
        inst_to_label = torch.load(os.path.join(embodied_base_dir, 'ScanNet','instance_id_to_label', f'{scan_id}.pth'))
        inst_to_text_label = inst_to_label.copy()
        gts[scan_id] = {}
        for obj_id in inst_to_label.keys():
            obj_607_name = inst_to_label[obj_id]
            obj_raw_id = label_converter.raw_name_to_scannet_raw_id[obj_607_name]
            obj_scannet_200_label = label_converter.scannet_raw_id_to_scannet200_id[obj_raw_id] if obj_raw_id in label_converter.scannet_raw_id_to_scannet200_id.keys() else -100
            if obj_scannet_200_label in [-100, 0, 2, 35]:
                continue
            obj_sem_feat = scannet_607_cat_to_text_embed[inst_to_text_label[obj_id]]
            gts[scan_id][obj_id] = {'class_id': obj_scannet_200_label, 'open_vocab_feat': obj_sem_feat}
    # eval
    class_correct = 0
    class_total = 0
    sem_sim_sum = 0
    correct_sim_sum = 0
    for scan_id in eval_scan_ids:
        for obj_id in preds[scan_id].keys():
            if obj_id not in gts[scan_id].keys():
                continue
            class_total += 1
            cur_sim = torch.cosine_similarity(torch.Tensor(preds[scan_id][obj_id]['open_vocab_feat']), gts[scan_id][obj_id]['open_vocab_feat'], dim=0)
            if preds[scan_id][obj_id]['class_id'] == gts[scan_id][obj_id]['class_id']:
                class_correct += 1
                correct_sim_sum += cur_sim
            sem_sim_sum += cur_sim
    # print result
    class_acc = class_correct / class_total
    sem_sim = sem_sim_sum / class_total
    correct_sem_sim = correct_sim_sum / class_correct
    incorrect_sem_sim = (sem_sim_sum - correct_sim_sum) / (class_total - class_correct)
    print(f"semantic class accuracy: {class_acc}, semantic similarity: {sem_sim}, correct semantic similarity: {correct_sem_sim}, incorrect semantic similarity: {incorrect_sem_sim}")
    return {'class_acc': class_acc, 'sem_sim': sem_sim}
        
# inputs preds {'scan_id': 'pred_masks': masks, 'pred_classes': classes, 'pred_scores': scores}
def eval_mask_scannet(preds, cfg): 
    # load meta data
    base_dir = cfg.data.scene_verse_base
    embodied_base_dir = cfg.data.embodied_base
    # load label converter
    int2cat = json.load(open(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
    cat2int = {w: i for i, w in enumerate(int2cat)}
    label_converter = LabelConverter(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2-labels.combined.tsv"))
    scannet_607_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "ScanNet", "scannet_sem_text_feature.pth"))
    hm3d_sem_category_mapping = np.loadtxt(os.path.join(embodied_base_dir, "HM3D", "hm3dsem_category_mappings.tsv"), dtype=str, delimiter="\t", encoding="utf-8")
    hm3d_raw_to_scannet607 = convert_gpt4
    hm3d_raw_to_cat = {hm3d_sem_category_mapping[i, 0]: hm3d_sem_category_mapping[i, 1] for i in range(1, hm3d_sem_category_mapping.shape[0])}
    hm3d_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "HM3D", "hm3d_sem_text_feature.pth")) 
    # build gt_ids
    gt_ids = {}
    valid_scan_ids = list(preds.keys())
    for scan_id in valid_scan_ids:
        instance_labels = np.fromfile(os.path.join(embodied_base_dir, 'ScanNet', 'instance_mask_global',  f'{scan_id}.bin'), dtype=np.int64).reshape(-1)
        instance_labels = instance_labels - 1
        instance_labels[instance_labels == -1] = -100
        inst_to_label = torch.load(os.path.join(embodied_base_dir, 'ScanNet','instance_id_to_label', f'{scan_id}.pth'))
        gt_mask_list = []
        for inst_id in inst_to_label.keys():
            if label_converter.raw_name_to_scannet_raw_id[inst_to_label[inst_id]] in label_converter.scannet_raw_id_to_scannet200_id and inst_to_label[inst_id] not in ['wall', 'floor', 'ceiling']:
                gt_mask_list.append(instance_labels == inst_id)
        preds[scan_id]['pred_classes'] = np.ones(len(preds[scan_id]['pred_classes']))
        cur_ids = np.zeros(preds[scan_id]['pred_masks'].shape[0])
        for i, mask in enumerate(gt_mask_list):
            cur_ids[mask] = 1 * 1000 + i + 1 # class * 1000 + object
        gt_ids[scan_id] = cur_ids
    # eval
    eval_results = eval_instseg_flexible(preds, gt_ids, ['object'])
    print(eval_results)
    return eval_results

# inputs preds {'scan_id': 'pred_masks': masks, 'pred_scores': scores, 'pred_openvocab': openvocab}
def eval_mask_open_vocab_scannet(preds, cfg): 
    # load meta data
    base_dir = cfg.data.scene_verse_base
    embodied_base_dir = cfg.data.embodied_base
    # load label converter
    int2cat = json.load(open(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
    cat2int = {w: i for i, w in enumerate(int2cat)}
    label_converter = LabelConverter(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2-labels.combined.tsv"))
    scannet_607_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "ScanNet", "scannet_sem_text_feature.pth"))
    hm3d_sem_category_mapping = np.loadtxt(os.path.join(embodied_base_dir, "HM3D", "hm3dsem_category_mappings.tsv"), dtype=str, delimiter="\t", encoding="utf-8")
    hm3d_raw_to_scannet607 = convert_gpt4
    hm3d_raw_to_cat = {hm3d_sem_category_mapping[i, 0]: hm3d_sem_category_mapping[i, 1] for i in range(1, hm3d_sem_category_mapping.shape[0])}
    hm3d_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "HM3D", "hm3d_sem_text_feature.pth")) 
    # build gt_ids
    gt_ids = {}
    valid_scan_ids = list(preds.keys())
    for scan_id in valid_scan_ids:
        instance_labels = np.fromfile(os.path.join(embodied_base_dir, 'ScanNet', 'instance_mask_global',  f'{scan_id}.bin'), dtype=np.int64).reshape(-1)
        instance_labels = instance_labels - 1
        instance_labels[instance_labels == -1] = -100
        inst_to_label = torch.load(os.path.join(embodied_base_dir, 'ScanNet','instance_id_to_label', f'{scan_id}.pth'))
        gt_mask_dict = defaultdict(list)
        for inst_id in inst_to_label.keys():
            if label_converter.raw_name_to_scannet_raw_id[inst_to_label[inst_id]] in label_converter.scannet_raw_id_to_scannet200_id and inst_to_label[inst_id] not in ['wall', 'floor', 'ceiling']:
                scannet200_label = label_converter.scannet_raw_id_to_scannet200_id[label_converter.raw_name_to_scannet_raw_id[inst_to_label[inst_id]]]
                gt_mask_dict[scannet200_label].append(instance_labels == inst_id)
        cur_ids = np.zeros(preds[scan_id]['pred_masks'].shape[0])
        for i, mask_list in gt_mask_dict.items():
            for j, mask in enumerate(mask_list):
                cur_ids[mask] = (i + 1) * 1000 + j + 1  # start from 1
        gt_ids[scan_id] = cur_ids
    # build pred classes
    scannet_607_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "ScanNet", "scannet_sem_text_feature.pth"))
    scannet_200_text_feat = []
    for i in range(200):
        scannet_200_text_feat.append(scannet_607_cat_to_text_embed[label_converter.scannet_raw_id_to_raw_name[VALID_CLASS_IDS_200[i]]])
    scannet_200_text_feat = torch.stack(scannet_200_text_feat, dim=0) # N, 607
    for scan_id in valid_scan_ids:
        pred_classes = np.zeros(len(preds[scan_id]['pred_openvocab']))
        for i in range(len(preds[scan_id]['pred_openvocab'])):
            cur_openvocab = torch.Tensor(preds[scan_id]['pred_openvocab'][i])
            cur_sim = torch.cosine_similarity(cur_openvocab, scannet_200_text_feat, dim=1)
            pred_classes[i] = torch.argmax(cur_sim).item() + 1
        preds[scan_id]['pred_classes'] = pred_classes
    # convert pred
    # eval
    eval_results = eval_instseg_flexible(preds, gt_ids, CLASS_LABELS_200)
    print(eval_results)
    return eval_results

def eval_mask_open_vocab_hm3d(preds, cfg): 
    # load meta data
    base_dir = cfg.data.scene_verse_base
    embodied_base_dir = cfg.data.embodied_base
    # load label converter
    int2cat = json.load(open(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
    cat2int = {w: i for i, w in enumerate(int2cat)}
    label_converter = LabelConverter(os.path.join(base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2-labels.combined.tsv"))
    scannet_607_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "ScanNet", "scannet_sem_text_feature.pth"))
    hm3d_sem_category_mapping = np.loadtxt(os.path.join(embodied_base_dir, "HM3D", "hm3dsem_category_mappings.tsv"), dtype=str, delimiter="\t", encoding="utf-8")
    hm3d_raw_to_scannet607 = convert_gpt4
    hm3d_raw_to_cat = {hm3d_sem_category_mapping[i, 0]: hm3d_sem_category_mapping[i, 1] for i in range(1, hm3d_sem_category_mapping.shape[0])}
    hm3d_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "HM3D", "hm3d_sem_text_feature.pth")) 
    # build gt_ids
    gt_ids = {}
    valid_scan_ids = list(preds.keys())
    for scan_id in valid_scan_ids:
        scan_id_ori = scan_id.split('_')[0]
        instance_labels = np.fromfile(os.path.join(embodied_base_dir, 'HM3D', 'instance_mask_global',  f'{scan_id}.bin'), dtype=np.int64).reshape(-1)
        inst_to_label = torch.load(os.path.join(embodied_base_dir, 'HM3D','instance_id_to_label', f'{scan_id_ori}_00.pth'))
        gt_mask_dict = defaultdict(list)
        for inst_id in inst_to_label.keys():
            if inst_to_label[inst_id] in convert_gpt4.keys() and  label_converter.raw_name_to_scannet_raw_id[convert_gpt4[inst_to_label[inst_id]]] in label_converter.scannet_raw_id_to_scannet200_id and convert_gpt4[inst_to_label[inst_id]] not in ['wall', 'floor', 'ceiling']:
                scannet200_label = label_converter.scannet_raw_id_to_scannet200_id[label_converter.raw_name_to_scannet_raw_id[convert_gpt4[inst_to_label[inst_id]]]]
                gt_mask_dict[scannet200_label].append(instance_labels == inst_id)
        cur_ids = np.zeros(preds[scan_id]['pred_masks'].shape[0])
        for i, mask_list in gt_mask_dict.items():
            for j, mask in enumerate(mask_list):
                cur_ids[mask] = (i + 1) * 1000 + j + 1  # start from 1
        gt_ids[scan_id] = cur_ids
    # build pred classes
    scannet_607_cat_to_text_embed = torch.load(os.path.join(embodied_base_dir, "ScanNet", "scannet_sem_text_feature.pth"))
    scannet_200_text_feat = []
    for i in range(200):
        scannet_200_text_feat.append(scannet_607_cat_to_text_embed[label_converter.scannet_raw_id_to_raw_name[VALID_CLASS_IDS_200[i]]])
    scannet_200_text_feat = torch.stack(scannet_200_text_feat, dim=0) # N, 607
    for scan_id in valid_scan_ids:
        pred_classes = np.zeros(len(preds[scan_id]['pred_openvocab']))
        for i in range(len(preds[scan_id]['pred_openvocab'])):
            cur_openvocab = torch.Tensor(preds[scan_id]['pred_openvocab'][i])
            cur_sim = torch.cosine_similarity(cur_openvocab, scannet_200_text_feat, dim=1)
            pred_classes[i] = torch.argmax(cur_sim).item() + 1
        preds[scan_id]['pred_classes'] = pred_classes
    # convert pred
    # eval
    eval_results = eval_instseg_flexible(preds, gt_ids, CLASS_LABELS_200)
    print(eval_results)
    return eval_results