import os
import collections
import json
import pickle
import random
import jsonlines
from torch_scatter import scatter_mean, scatter_add
from tqdm import tqdm
from scipy import sparse
import numpy as np
import torch
from torch.utils.data import Dataset

from common.misc import rgetattr
from ..data_utils import convert_pc_to_box, LabelConverter, build_rotate_mat, load_matrix_from_txt, \
                        construct_bbox_corners, eval_ref_one_sample  
from copy import deepcopy


SCAN_DATA = {}

class ScanNetBase(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.base_dir = cfg.data.scan_family_base
        self.load_scan_options = self.cfg.data.get('load_scan_options', {})
        assert self.split in ['train', 'val', 'test']

        self.int2cat = json.load(open(os.path.join(self.base_dir,
                                            "annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(self.base_dir,
                                            "annotations/meta_data/scannetv2-labels.combined.tsv"))

    def __len__(self):
        return len(self.lang_data)

    def __getitem__(self, index):
        scan_id, tgt_object_id, tgt_object_name, sentence, data_dict = self.get_lang(index)
        scene_dict = self.get_scene(scan_id, tgt_object_id, tgt_object_name, sentence)
        data_dict.update(scene_dict)
        return data_dict
    
    def match_gt_to_pred(self, scan_data):
        gt_boxes = scan_data['obj_box']
        pred_boxes = scan_data['obj_box_pred']

        def get_iou_from_box(gt_box, pred_box):
            gt_corners = construct_bbox_corners(gt_box[:3], gt_box[3:6])
            pred_corners = construct_bbox_corners(pred_box[:3], pred_box[3:6])
            iou = eval_ref_one_sample(gt_corners, pred_corners)
            return iou
        
        matched_list, iou25_list, iou50_list = [], [], []
        for cur_id in range(len(gt_boxes)):
            gt_box = gt_boxes[cur_id]
            max_iou = -1
            iou25, iou50 = [], []
            for i in range(len(pred_boxes)):
                pred_box = pred_boxes[i]
                iou = get_iou_from_box(gt_box, pred_box)
                if iou > max_iou:
                    max_iou = iou
                    object_id_matched = i
                # find tgt iou 25
                if iou >= 0.25:
                    iou25.append(i)
                # find tgt iou 50
                if iou >= 0.5:
                    iou50.append(i)
            matched_list.append(object_id_matched)
            iou25_list.append(iou25)
            iou50_list.append(iou50)
        
        scan_data['matched_list'] = matched_list
        scan_data['iou25_list'] = iou25_list
        scan_data['iou50_list'] = iou50_list

    def _load_one_scan(self, scan_id):
        options = self.load_scan_options
        one_scan = {}
        if options.get('load_inst_info', True):
            inst_labels, inst_locs, inst_colors = self._load_inst_info(scan_id)
            one_scan['inst_labels'] = inst_labels # (n_obj, )
            one_scan['inst_locs'] = inst_locs # (n_obj, 6) center xyz, whl
            one_scan['inst_colors'] = inst_colors # (n_obj, 3x4) cluster * (weight, mean rgb)

        if options.get('load_pc_info', True):
            # load pcd data
            pcd_data = torch.load(os.path.join(self.base_dir, "scan_data",
                                            "pcd_with_global_alignment", f'{scan_id}.pth'))
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            one_scan['pcds'] = pcds
            # convert to gt object
            if options.get('load_inst_info', True):

                def get_loc(obj_pcd):
                    center = obj_pcd[:, :3].mean(0)
                    size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                    return list(center) + list(size)
                
                def get_box(obj_pcd):
                    center, size = convert_pc_to_box(obj_pcd)
                    return list(center) + list(size)
                
                # gt masks
                obj_masks = instance_labels[None, :] == np.arange(instance_labels.max() + 1)[:, None] if instance_labels is not None else [] # ScanQA test
                obj_boxes = np.array([get_box(pcds[mask]) for mask in obj_masks], dtype=np.float32)
                obj_locs = np.array([get_loc(pcds[mask]) for mask in obj_masks], dtype=np.float32)
                bg_obj_ids = [i for i, label in enumerate(inst_labels) if self.int2cat[label] in ['wall', 'floor', 'ceiling']]
                bg_mask = [] if obj_masks == [] else obj_masks[bg_obj_ids].sum(0).astype(bool) # ScanQA test
                one_scan['obj_masks'] = [np.nonzero(mask)[0] for mask in obj_masks]
                one_scan['bg_mask'] = bg_mask
                one_scan['obj_loc'] = obj_locs
                one_scan['obj_box'] = obj_boxes

                obj_mask_path = os.path.join(self.base_dir, "mask", str(scan_id) + ".mask" + ".npz")
                if os.path.exists(obj_mask_path):
                    obj_label_path = os.path.join(self.base_dir, "mask", str(scan_id) + ".label" + ".npy")

                    topk = 50
                    obj_masks = np.array(sparse.load_npz(obj_mask_path).todense(), dtype=bool)[:topk] # TODO: only keep the top-50 bbox for now
                    inst_labels = np.load(obj_label_path)[:topk]
                    valid = obj_masks.any(axis=1)
                    obj_masks = obj_masks[valid]
                    inst_labels = inst_labels[valid]
                    obj_boxes = np.array([get_box(pcds[mask]) for mask in obj_masks], dtype=np.float32)
                    obj_locs = np.array([get_loc(pcds[mask]) for mask in obj_masks], dtype=np.float32)
                    bg_mask = ~(obj_masks.sum(axis=0) > 0)
                    one_scan['obj_masks_pred'] = [np.nonzero(mask)[0] for mask in obj_masks]
                    one_scan['bg_mask_pred'] = bg_mask
                    one_scan['obj_loc_pred'] = obj_locs
                    one_scan['obj_box_pred'] = obj_boxes
                    one_scan['inst_labels_pred'] = inst_labels
                    self.match_gt_to_pred(one_scan)
                
        if options.get('load_voxel_feats', False):
            one_scan['voxel_feats'] = self._load_voxel_feats(scan_id)
            one_scan['voxel_feats_pred'] = self._load_voxel_feats(scan_id, gt=False)

        if options.get('load_mv_feats', False):
            one_scan['mv_feats'] = self._load_mv_feats(scan_id)
            one_scan['mv_feats_pred'] = self._load_mv_feats(scan_id, gt=False)
        
        # load segment for mask3d
        if options.get('load_segment_info', False):
            one_scan["scene_pcds"] = np.load(os.path.join(self.base_dir, "scan_data", "pcd_mask3d", f'{scan_id[-7:]}.npy'))
        
        # load offline feature 
        if options.get('load_offline_segment_voxel', False):
            one_scan['offline_segment_voxel'] = torch.load(os.path.join(self.base_dir, "scan_data", "mask3d_voxel_feature", f'{scan_id}.pth'), map_location='cpu')
            
        if options.get('load_offline_segment_image', False):
            one_scan['offline_segment_image'] = torch.load(os.path.join(self.base_dir, "scan_data", "mask3d_image_feature", f'{scan_id}.pth'), map_location='cpu')
            
        if options.get('load_offline_segment_point', False):
            one_scan['offline_segment_point'] = torch.load(os.path.join(self.base_dir, "scan_data", "mask3d_point_feature", f'{scan_id}.pth'), map_location='cpu')

        return (scan_id, one_scan)

    def _load_scannet(self, scan_ids):
        process_num = self.load_scan_options.get('process_num', 0)
        unloaded_scan_ids = [scan_id for scan_id in scan_ids if scan_id not in SCAN_DATA]
        print(f"Loading scans: {len(unloaded_scan_ids)} / {len(scan_ids)}")
        scans = {}
        if process_num > 1:
            from joblib import Parallel, delayed
            res_all = Parallel(n_jobs=process_num)(
                delayed(self._load_one_scan)(scan_id) for scan_id in tqdm(unloaded_scan_ids))
            for scan_id, one_scan in tqdm(res_all):
                scans[scan_id] = one_scan
        else:
            for scan_id in tqdm(unloaded_scan_ids):
                _, one_scan = self._load_one_scan(scan_id)
                scans[scan_id] = one_scan

        SCAN_DATA.update(scans)
        scans = {scan_id: SCAN_DATA[scan_id] for scan_id in scan_ids}
        return scans

    def _load_lang(self, cfg):
        caption_source = cfg.sources
        lang_data = []
        if caption_source:
            if 'scanrefer' in caption_source:
                anno_file = os.path.join(self.base_dir, 'annotations/refer/scanrefer.jsonl')
                with jsonlines.open(anno_file, 'r') as _f:
                    for item in _f:
                        if item['scan_id'] in self.scannet_scan_ids:
                            lang_data.append(('scannet', item['scan_id'], item['utterance']))

            if 'referit3d' in caption_source:
                for anno_type in cfg.referit3d.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/{anno_type}.jsonl')
                    with jsonlines.open(anno_file, 'r') as _f:
                        for item in _f:
                            if item['scan_id'] in self.scannet_scan_ids:
                                lang_data.append(('scannet', item['scan_id'], item['utterance']))

            if 'scanqa' in caption_source:
                anno_file_list = ['annotations/qa/ScanQA_v1.0_train.json',
                                  'annotations/qa/ScanQA_v1.0_val.json']
                for anno_file in anno_file_list:
                    anno_file = os.path.join(self.base_dir, anno_file)
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scene_id'] in self.scannet_scan_ids:
                            for i in range(len(item['answers'])):
                                lang_data.append(('scannet', item['scene_id'],
                                                  item['question'] + " " + item['answers'][i]))

            if 'sgrefer' in caption_source:
                for anno_type in cfg.sgrefer.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_ref_{anno_type}.json')
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scan_id'] in self.scannet_scan_ids:
                            lang_data.append(('scannet', item['scan_id'], item['utterance']))

            if 'sgcaption' in caption_source:
                for anno_type in cfg.sgcaption.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_caption_{anno_type}.json')
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scan_id'] in self.scannet_scan_ids:
                            lang_data.append(('scannet', item['scan_id'], item['utterance']))
        return lang_data
    
    def _load_voxel_feats(self, scan_id, gt = True):
        mask_type = '_GT' if gt else ''
        feat_pth = os.path.join(self.base_dir, 'mask3d_obj_centric_feats' + mask_type, scan_id + '.pth')
        if not os.path.exists(feat_pth):
            return None
        feat_dict = torch.load(feat_pth)
        feat_dim = next(iter(feat_dict.values())).shape[0]
        n_obj = max(feat_dict.keys()) + 1 # the last one is for missing objects.
        feat = torch.zeros((n_obj, feat_dim), dtype=torch.float32)
        for k, v in feat_dict.items():
            feat[k] = v

        return feat
    
    def _load_mv_feats(self, scan_id, gt = True):
        mask_type = 'gt' if gt else 'pred'
        feat_pth = os.path.join(self.base_dir, f'openseg_obj_feats_{mask_type}_mask', scan_id + '.pt')        
        if not os.path.exists(feat_pth):
            return None
        feat = torch.load(feat_pth)
        feat = torch.cat((feat, torch.zeros_like(feat[:1])), dim=0) # the last one is for missing objects.
        return feat

    def _load_split(self, cfg, split, use_multi_process = False):
        if use_multi_process and split in ['train']:
            split_file = os.path.join(self.base_dir, 'annotations/splits/scannetv2_'+ split + "_sort.json")
            with open(split_file, 'r') as f:
                scannet_scan_ids = json.load(f)
        else:
            split_file = os.path.join(self.base_dir, 'annotations/splits/scannetv2_'+ split + ".txt")
            scannet_scan_ids = {x.strip() for x in open(split_file, 'r', encoding="utf-8")}
            scannet_scan_ids = sorted(scannet_scan_ids)

        if cfg.debug.flag and cfg.debug.debug_size != -1:
            scannet_scan_ids = list(scannet_scan_ids)[:cfg.debug.debug_size]
        return scannet_scan_ids

    def _load_inst_info(self, scan_id):
        if not os.path.exists(os.path.join(self.base_dir, 'scan_data', 'instance_id_to_name', f'{scan_id}.json')): # ScanQA test
            return [], [], []
        inst_labels = json.load(open(os.path.join(self.base_dir, 'scan_data',
                                                    'instance_id_to_name',
                                                    f'{scan_id}.json'), encoding="utf-8"))
        inst_labels = [self.cat2int[i] for i in inst_labels]

        inst_locs = np.load(os.path.join(self.base_dir, 'scan_data',
                                            'instance_id_to_loc', f'{scan_id}.npy'))
        inst_colors = json.load(open(os.path.join(self.base_dir, 'scan_data',
                                                    'instance_id_to_gmm_color',
                                                    f'{scan_id}.json'), encoding="utf-8"))
        inst_colors = [np.concatenate(
            [np.array(x['weights'])[:, None], np.array(x['means'])],
            axis=1).astype(np.float32) for x in inst_colors]

        return inst_labels, inst_locs, inst_colors

    def _obj_processing_post(self, obj_pcds, rot_aug=True):
        obj_pcds = torch.from_numpy(obj_pcds)
        rot_matrix = build_rotate_mat(self.split, rot_aug)
        if rot_matrix is not None:
            rot_matrix = torch.from_numpy(rot_matrix.transpose())
            obj_pcds[:, :, :3] @= rot_matrix
        
        xyz = obj_pcds[:, :, :3]
        center = xyz.mean(1)
        xyz_min = xyz.min(1).values
        xyz_max = xyz.max(1).values
        box_center = (xyz_min + xyz_max) / 2
        size = xyz_max - xyz_min
        obj_locs = torch.cat([center, size], dim=1)
        obj_boxes = torch.cat([box_center, size], dim=1)

        # centering
        obj_pcds[:, :, :3].sub_(obj_pcds[:, :, :3].mean(1, keepdim=True))

        # normalization
        max_dist = (obj_pcds[:, :, :3]**2).sum(2).sqrt().max(1).values
        max_dist.clamp_(min=1e-6)
        obj_pcds[:, :, :3].div_(max_dist[:, None, None])
        
        return obj_pcds, obj_locs, obj_boxes, rot_matrix

    def _obj_processing_aug(self, obj_pcds, obj_labels, is_need_bbox=False):
        # calculate size
        if self.augmentor:
            data_dict = self.augmentor.forward({'obj_pcds': obj_pcds,
                                                'num_points': self.num_points}
                                               )
        obj_pcds = data_dict['obj_pcds']
        obj_sizes = data_dict['obj_sizes']
        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for _i, obj_pcd in enumerate(obj_pcds):
            # build locs
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_sizes[_i]
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # build box
            if is_need_bbox:
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_sizes[_i]
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))

            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_fts.shape[0] == obj_locs.shape[0]

        return obj_fts, obj_locs, obj_boxes, obj_labels
    
    def _scene_processing_aug(self, obj_pcds, bg_pcds, obj_labels, is_need_bbox=False):
        # sample background points
        fg_points_num = len(obj_pcds) * self.num_points
        assert fg_points_num < self.max_pcd_num_points
        bg_points_num = min(self.max_pcd_num_points - fg_points_num, self.bg_points_num)
        assert len(bg_pcds) > 0
        assert bg_points_num > 0
        bg_points_indices = np.random.choice(len(bg_pcds), size=bg_points_num, replace=len(bg_pcds) < bg_points_num)
        bg_pcds = bg_pcds[bg_points_indices]

        # augment objects
        if self.augmentor:
            data_dict = self.augmentor.forward({'obj_pcds': obj_pcds, 
                                                'bg_pcds': bg_pcds, 
                                                'num_points': self.num_points})

        obj_pcds = data_dict['obj_pcds']
        obj_sizes = data_dict['obj_sizes']
        bg_pcds = data_dict['bg_pcds']
        assert len(obj_pcds) * obj_pcds[0].shape[0] == fg_points_num

        # calculate location and generate scene_pcd
        obj_locs = []
        obj_boxes = []
        scene_pcds = []
        for _i, obj_pcd in enumerate(obj_pcds):
            # build locs
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_sizes[_i]
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # build box
            if is_need_bbox:
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_sizes[_i]
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))

            # build scene pcd
            scene_pcds.extend(obj_pcd)

        # # sample background points
        # assert len(scene_pcds) < self.max_pcd_num_points
        # bg_points_num = min(self.max_pcd_num_points - len(scene_pcds), self.bg_points_num)
        # assert len(bg_pcds) > 0
        # assert bg_points_num > 0
        # bg_points_indices = np.random.choice(len(bg_pcds), size=bg_points_num, replace=len(bg_pcds) < bg_points_num)
        # scene_pcds.extend(bg_pcds[bg_points_indices])
        # assert len(scene_pcds) == self.max_pcd_num_points

        scene_pcds.extend(bg_pcds)

        # generate obj point indices masks
        obj_pcds_masks = []
        offset = 0
        for _j in range(len(obj_pcds)):
            mask = np.arange(self.num_points) + offset
            assert len(mask) == len(obj_pcds[_j])
            obj_pcds_masks.append(mask)
            offset += self.num_points

        # convert to torch
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)
        obj_pcds_masks = torch.from_numpy(np.array(obj_pcds_masks))
        scene_pcds = np.array(scene_pcds)

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_pcds_masks.shape[0] == obj_locs.shape[0]

        return obj_locs, obj_boxes, obj_labels, obj_pcds_masks, scene_pcds

    def init_dataset_params(self, dataset_cfg):
        if dataset_cfg is None:
            dataset_cfg = {}
        self.pc_type = dataset_cfg.get('pc_type', 'gt')
        self.sem_type = dataset_cfg.get('sem_type', '607')
        self.max_obj_len = dataset_cfg.get('max_obj_len', 80)
        self.num_points = dataset_cfg.get('num_points', 1024)
        self.filter_lang = dataset_cfg.get('filter_lang', False)
        self.rot_aug = dataset_cfg.get('rot_aug', True)
        self.train_duplicate = dataset_cfg.get('train_duplicate', 1)
        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']

    def init_scan_data(self):
        self.scan_data = self._load_scannet(self.scan_ids)

        # build unique multiple look up
        if self.load_scan_options.get('load_inst_info', True):
            for scan_id in self.scan_ids:
                inst_labels = self.scan_data[scan_id]['inst_labels']
                self.scan_data[scan_id]['label_count'] = collections.Counter(
                                    [self.label_converter.id_to_scannetid[l] for l in inst_labels])
        
        if self.load_scan_options.get('load_segment_center', False):
            for scan_id in self.scan_ids:
                points = deepcopy(self.scan_data[scan_id]['scene_pcds']) # mask3d points
                coordinates, color, normals, point2seg_id, labels = (
                    points[:, :3],
                    points[:, 3:6],
                    points[:, 6:9],
                    points[:, 9],
                    points[:, 10:12]  if points.shape[1] > 10 else np.ones((points.shape[0], 2)) * 2,
                )
                coordinates = torch.from_numpy(coordinates).float()
                point2seg_id = torch.from_numpy(point2seg_id).long()
                seg_center = scatter_mean(coordinates, point2seg_id, dim=0)
                self.scan_data[scan_id]['seg_center'] = seg_center
        
        if self.load_scan_options.get('load_segment_mask', False):
           for scan_id in self.scan_ids:
                labels = deepcopy(self.scan_data[scan_id]['scene_pcds'])[:, 9:12]
                # get ids
                point2seg_id, point2inst_label, point2inst_id = labels[:, 0], labels[:, 1], labels[:, 2]
                point2inst_label = self.map_to_scannet200_id(point2inst_label)
                point2seg_id = point2seg_id.astype(int)
                point2inst_label = point2inst_label.astype(int)
                point2inst_id = point2inst_id.astype(int)
                # build gt masks
                if 'obj_masks' in self.scan_data[scan_id].keys():
                    unique_inst_ids, indices, inverse_indices = np.unique(point2inst_id, return_index=True, return_inverse=True)
                    n_inst = len(unique_inst_ids)
                    # instance to segment mask
                    n_segments = point2seg_id.max() + 1
                    segment_masks = np.zeros((n_inst, n_segments), dtype=bool)
                    segment_masks[inverse_indices, point2seg_id] = True
                    # unique labels
                    unique_inst_labels = point2inst_label[indices]
                    # filter out unwanted instances
                    valid = (unique_inst_ids != -1)
                    unique_inst_ids = unique_inst_ids[valid]
                    unique_inst_labels = unique_inst_labels[valid]
                    segment_masks = segment_masks[valid]
                    segment_masks = torch.from_numpy(segment_masks)
                    instance_labels = torch.from_numpy(unique_inst_labels)
                    assert segment_masks.shape[0] == len(self.scan_data[scan_id]['obj_masks'])
                    assert segment_masks.shape[0] == len(unique_inst_labels)
                    self.scan_data[scan_id]['segment_masks_gt'] = segment_masks
                    self.scan_data[scan_id]['instance_labels_gt'] = instance_labels
                # build pred mask
                obj_mask_path = os.path.join(self.base_dir, "mask", str(scan_id) + ".mask" + ".npz")
                if os.path.exists(obj_mask_path):
                    topk = 50
                    obj_masks = np.array(sparse.load_npz(obj_mask_path).todense(), dtype=bool)[:topk] # TODO: only keep the top-50 bbox for now
                    valid = obj_masks.any(axis=1)
                    obj_masks = obj_masks[valid]
                    n_inst = obj_masks.shape[0]
                    segment_masks = scatter_add(torch.from_numpy(obj_masks), torch.from_numpy(point2seg_id), dim=1) > 0
                    assert segment_masks.shape[0] == len(self.scan_data[scan_id]['obj_masks_pred'])
                    self.scan_data[scan_id]['segment_masks_pred'] = segment_masks
                    self.scan_data[scan_id]['instance_labels_pred'] = torch.ones((segment_masks.shape[0]))
    
    def map_to_scannet200_id(self, labels):
        label_info = self.label_converter.scannet_raw_id_to_scannet200_id
        labels[~np.isin(labels, list(label_info))] = -100
        for k in label_info:
            labels[labels == k] = label_info[k]
        return labels
    
    def get_lang(self, index):
        raise NotImplementedError

    def get_scene(self, scan_id, tgt_object_id_list, tgt_object_name_list, sentence):
        if not isinstance(tgt_object_id_list, list):
            tgt_object_id_list = [tgt_object_id_list]
        if not isinstance(tgt_object_name_list, list):
            tgt_object_name_list = [tgt_object_name_list]

        scan_data = self.scan_data[scan_id]
        pcds = deepcopy(scan_data['pcds'])

        # gt masks
        obj_masks = scan_data['obj_masks']
        obj_boxes = scan_data['obj_box']
        obj_locs = scan_data['obj_loc']
        obj_labels = scan_data['inst_labels'] # N

        tgt_obj_boxes = obj_boxes[tgt_object_id_list]
        tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]
        is_multiple = sum([scan_data['label_count'][self.label_converter.id_to_scannetid[x]] 
                           for x in tgt_object_label_list]) > 1
        tgt_object_id_iou25_list = tgt_object_id_list
        tgt_object_id_iou50_list = tgt_object_id_list

        # pred masks
        if self.pc_type == 'pred':
            obj_masks = scan_data['obj_masks_pred']
            obj_boxes = scan_data['obj_box_pred']
            obj_locs = scan_data['obj_loc_pred']
            obj_labels = scan_data['inst_labels_pred']

            # rebulid tgt_object_id_list
            iou_list = [j for i in tgt_object_id_list for j in scan_data['iou25_list'][i]]
            tgt_object_id_iou25_list = sorted(set(iou_list))
            iou_list = [j for i in tgt_object_id_list for j in scan_data['iou50_list'][i]]
            tgt_object_id_iou50_list = sorted(set(iou_list))
            tgt_object_id_list = [scan_data['matched_list'][i] for i in tgt_object_id_list]
            tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]

        excluded_labels = ['wall', 'floor', 'ceiling']
        def keep_obj(i, obj_label):
            # do not filter for predicted labels, because these labels are not accurate
            if self.pc_type != 'gt' or i in tgt_object_id_list:
                return True
            category = self.int2cat[obj_label]
            # filter out background
            if category in excluded_labels:
                return False
            # filter out objects not mentioned in the sentence
            if self.filter_lang and category not in sentence:
                return False
            return True
        selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if keep_obj(i, obj_label)]

        # crop objects to max_obj_len
        if self.max_obj_len < len(selected_obj_idxs):
            pre_selected_obj_idxs = selected_obj_idxs
            # select target first
            selected_obj_idxs = tgt_object_id_list[:]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for i in pre_selected_obj_idxs:
                label = obj_labels[i]
                if i not in selected_obj_idxs:
                    if label in tgt_object_label_list:
                        selected_obj_idxs.append(i)
                    else:
                        remained_obj_idx.append(i)
                if len(selected_obj_idxs) >= self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # assert len(selected_obj_idxs) == self.max_obj_len

        # reorganize ids
        tgt_object_id_list = [selected_obj_idxs.index(id) for id in tgt_object_id_list]
        tgt_object_id_iou25_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou25_list]
        tgt_object_id_iou50_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou50_list]

        obj_labels = [obj_labels[i] for i in selected_obj_idxs]
        obj_masks = [obj_masks[i] for i in selected_obj_idxs]
        # subsample points
        selected_points = [
            np.random.choice(indices, size=self.num_points, replace=len(indices) < self.num_points)
            for indices in obj_masks
        ]
        selected_points = np.stack(selected_points)
        obj_pcds = pcds[selected_points]

        obj_fts, obj_locs, obj_boxes, rot_matrix = self._obj_processing_post(obj_pcds, rot_aug=self.rot_aug)

        data_dict = {
            "scan_id": scan_id,
            "tgt_object_id": torch.LongTensor(tgt_object_id_list),
            "tgt_object_label": torch.LongTensor(tgt_object_label_list),
            "tgt_obj_boxes": tgt_obj_boxes.tolist(), # w/o augmentation, only use it for evaluation, tolist is for collate.
            "obj_fts": obj_fts.float(),
            "obj_locs": obj_locs.float(),
            "obj_labels": torch.LongTensor(obj_labels),
            "obj_boxes": obj_boxes, 
            "obj_pad_masks": torch.ones((len(obj_locs)), dtype=torch.bool), # used for padding in collate
            "tgt_object_id_iou25": torch.LongTensor(tgt_object_id_iou25_list),
            "tgt_object_id_iou50": torch.LongTensor(tgt_object_id_iou50_list), 
            'is_multiple': is_multiple
        }

        feat_type = '' if self.pc_type == 'gt' else '_pred'
        if self.load_scan_options.get('load_mv_feats', False):
            feats = scan_data['mv_feats' + feat_type]
            valid = [i if i < len(feats) else -1 for i in selected_obj_idxs]
            data_dict['mv_seg_fts'] = feats[valid]
            data_dict['mv_seg_pad_masks'] = torch.ones(len(data_dict['mv_seg_fts']), dtype=torch.bool)

        if self.load_scan_options.get('load_voxel_feats', False):
            feats = scan_data['voxel_feats' + feat_type]
            valid = [i if i < len(feats) else -1 for i in selected_obj_idxs]
            data_dict['voxel_seg_fts'] = feats[valid]
            data_dict['voxel_seg_pad_masks'] = torch.ones(len(data_dict['voxel_seg_fts']), dtype=torch.bool)
        
        data_dict['pc_seg_fts'] = obj_fts.float()
        data_dict['pc_seg_pad_masks'] = torch.ones(len(data_dict['pc_seg_fts']), dtype=torch.bool)
        
        if self.load_scan_options.get('load_offline_segment_voxel', False):
            feats = deepcopy(scan_data['offline_segment_voxel'])
            data_dict['voxel_seg_fts'] = feats['voxel_seg_feature']
            data_dict['voxel_seg_pad_masks'] = torch.ones(len(feats['voxel_seg_feature'][0]), dtype=torch.bool)
            
        if self.load_scan_options.get('load_offline_segment_image', False):
            feats = deepcopy(scan_data['offline_segment_image'])
            data_dict['mv_seg_fts'] = feats['image_seg_feature']
            data_dict['mv_seg_pad_masks'] = feats['image_seg_mask'].logical_not()
        
        if self.load_scan_options.get('load_offline_segment_point', False):
            feats = deepcopy(scan_data['offline_segment_point'])
            data_dict['pc_seg_fts'] = feats['point_seg_feature']
            data_dict['pc_seg_pad_masks'] = feats['point_seg_mask'].logical_not()
        
        data_dict['seg_center'] = obj_locs.float()
        data_dict['seg_pad_masks'] = data_dict['obj_pad_masks']
        
        if self.load_scan_options.get('load_segment_center', False):
            seg_center = deepcopy(scan_data['seg_center'])
            if rot_matrix is not None:
                seg_center @= rot_matrix
            data_dict['seg_center'] = seg_center
            data_dict['seg_pad_masks'] =  torch.ones(len(seg_center), dtype=torch.bool)
        
        if self.load_scan_options.get('load_segment_mask', False):
            segment_masks = deepcopy(scan_data[f'segment_masks_{self.pc_type}'])
            instance_labels = deepcopy(scan_data[f'instance_labels_{self.pc_type}'])
            data_dict['segment_masks'] = segment_masks[selected_obj_idxs]
            data_dict['instance_labels'] = instance_labels[selected_obj_idxs]
            assert data_dict['segment_masks'].shape[0] == data_dict['obj_locs'].shape[0]
        
        if self.load_scan_options.get('load_query_info', False):
            data_dict['query_locs'] = data_dict['obj_locs'].clone()
            data_dict['query_pad_masks'] = data_dict['obj_pad_masks'].clone()
            data_dict['coord_min'] = data_dict['obj_locs'][:, :3].min(0)[0]
            data_dict['coord_max'] = data_dict['obj_locs'][:, :3].max(0)[0]
            
        return data_dict