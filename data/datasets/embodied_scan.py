import os
import json
import random
import pickle
import jsonlines
import collections
from copy import deepcopy
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_scatter import scatter_mean, scatter_add, scatter_min, scatter_max
from tqdm import tqdm
from scipy import sparse
import volumentations as V
import albumentations as A
import MinkowskiEngine as ME
from transformers import AutoTokenizer

from common.misc import rgetattr
from ..data_utils import (
    convert_pc_to_box, LabelConverter, build_rotate_mat, load_matrix_from_txt,
    construct_bbox_corners, eval_ref_one_sample
)
from data.build import DATASET_REGISTRY
from data.datasets.constant import CLASS_LABELS_200, PromptType
from data.data_utils import make_bce_label
from data.datasets.hm3d_label_convert import convert_gpt4
import fpsample

SCAN_DATA = {'ScanNet': {}, 'HM3D': {}}

'''
Usage of this class
init_dataset_params() use dataset options
init()
load_split()
init_scan_data() use load_scan_options
'''
class EmbodiedScanBase(Dataset, ABC):
    def __init__(self, cfg, dataset_name, split) -> None:
        # basic settings
        self.cfg = cfg
        self.dataset_name = dataset_name # ['ScanNet']
        assert split in ['train', 'val']
        self.split = split
        self.base_dir = cfg.data.scene_verse_base
        self.embodied_base_dir = cfg.data.embodied_base
        self.load_scan_options = cfg.data.get('load_scan_options', {})
        # label converter for scannet
        self.int2cat = json.load(open(os.path.join(self.base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(self.base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2-labels.combined.tsv"))
        self.scannet_607_cat_to_text_embed = torch.load(os.path.join(self.embodied_base_dir, "ScanNet", "scannet_sem_text_feature.pth"))
        # label converter for hm3d
        hm3d_sem_category_mapping = np.loadtxt(os.path.join(self.embodied_base_dir, "HM3D", "hm3dsem_category_mappings.tsv"), dtype=str, delimiter="\t", encoding='utf-8')
        self.hm3d_raw_to_scannet607 = convert_gpt4
        self.hm3d_raw_to_cat = {hm3d_sem_category_mapping[i, 0]: hm3d_sem_category_mapping[i, 1] for i in range(1, hm3d_sem_category_mapping.shape[0])}
        self.hm3d_cat_to_text_embed = torch.load(os.path.join(self.embodied_base_dir, "HM3D", "hm3d_sem_text_feature.pth"))
         
    def __len__(self):
        return len(self.lang_data)
    
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("This is an abstract class.")
    
    # init datasets
    def init_dataset_params(self, dataset_cfg):
        if dataset_cfg is None:
            dataset_cfg = {}
        self.train_duplicate = dataset_cfg.get('train_duplicate', 1)
        self.load_frame_interval = dataset_cfg.get('load_frame_interval', 1)
        self.val_load_scan_max_num = dataset_cfg.get('val_load_scan_max_num', -1)

    def _load_split(self, cfg, split):
        if self.dataset_name == 'ScanNet':
            # scannet scan ids, scene0000_00,..
            if split == 'train':
                split_file = os.path.join(self.base_dir, 'ScanNet/annotations/splits/scannetv2_' + split + "_sort.json")
                with open(split_file, 'r') as f:
                    scan_ids = json.load(f)
            else:
                split_file = os.path.join(self.base_dir, 'ScanNet/annotations/splits/scannetv2_' + split + ".txt")
                scan_ids = {x.strip() for x in open(split_file, 'r', encoding="utf-8")}
                scan_ids = sorted(scan_ids)
        elif self.dataset_name == 'HM3D':
            # hm3d scan ids 000853-XUdsaknjsa,..., 
            train_val_split = json.load(open(os.path.join(self.embodied_base_dir, 'HM3D', 'hm3d_annotated_basis.scene_dataset_config.json')))
            scan_ids = [pa.split('/')[1] for pa in train_val_split['scene_instances']['paths']['.json'] if pa.startswith(split)]
            # add sub trajectory to scan id, the scan_id we save in one_scan is actually scan_id + sub_trajectory_id
            all_sub_trajectory_ids = os.listdir(os.path.join(self.embodied_base_dir, 'HM3D', 'points'))
            selected_sub_trajectory_ids = []
            for scan_id in scan_ids:
                cur_sub_trajecotry = [sub_id for sub_id in all_sub_trajectory_ids if sub_id.startswith(scan_id)]
                assert len(cur_sub_trajecotry) > 0, f'scan id {scan_id} not found'
                selected_sub_trajectory_ids += cur_sub_trajecotry
            scan_ids = selected_sub_trajectory_ids
            scan_ids = sorted(scan_ids)
        else:
            raise NotImplemented(f'data set name {self.dataset_name}')
        
        if self.split == 'val' and self.val_load_scan_max_num != -1:
            scan_ids = list(scan_ids)[:self.val_load_scan_max_num]
        
        if cfg.debug.flag and cfg.debug.debug_size != -1:
            scan_ids = list(scan_ids)[:cfg.debug.debug_size]
        
        return scan_ids
        
    def init_scan_data(self):
        self.scan_data = self._load_scans(self.scan_ids)
        
    def _load_scans(self, scan_ids):
        process_num = self.load_scan_options.get('process_num', 0)
        unloaded_scan_ids = [scan_id for scan_id in scan_ids if scan_id not in SCAN_DATA[self.dataset_name]]
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

        SCAN_DATA[self.dataset_name].update(scans)
        scans = {scan_id: SCAN_DATA[self.dataset_name][scan_id] for scan_id in scan_ids}
        return scans
    
    def _load_one_scan(self, scan_id):
        options = self.load_scan_options
        one_scan = {'sub_frames': {}}
        # get sub frame list
        if self.dataset_name == 'ScanNet':
            sub_frame_list = os.listdir(os.path.join(self.embodied_base_dir, 'ScanNet', 'points', scan_id))
        elif self.dataset_name == 'HM3D':
            sub_frame_list = []
            meta_path = os.path.join(self.embodied_base_dir, 'HM3D', 'meta', scan_id + ".json")
            meta_info = json.load(open(meta_path))
            sub_frame_list = []
            for sub_frame_id, ratio in meta_info.items():
                if ratio < options.get('max_invalid_point_ratio', 5):
                    sub_frame_list.append(sub_frame_id + ".bin")
        # get inst_to_label
        if self.dataset_name == 'ScanNet':
            inst_to_label = torch.load(os.path.join(self.base_dir, 'ScanNet', 'scan_data/instance_id_to_label', f'{scan_id}.pth')) 
            inst_to_text_label = inst_to_label.copy()
        else:
            # k - 1 because hm3d instance id starts from 1, 0 is no object
            scan_id_ori = scan_id.split('_')[0]
            inst_to_label = torch.load(os.path.join(self.embodied_base_dir, 'HM3D','instance_id_to_label', f'{scan_id_ori}_00.pth'))
            inst_to_text_label = {k - 1: self.hm3d_raw_to_cat[v] if v in self.hm3d_raw_to_cat.keys() else v for k,v in inst_to_label.items()}
            inst_to_label = {k - 1: self.hm3d_raw_to_scannet607[v] if v in self.hm3d_raw_to_scannet607.keys() else 'object' for k, v in inst_to_label.items()}
        one_scan['inst_to_label'] = inst_to_label
        one_scan['inst_to_text_label'] = inst_to_text_label
        # frames
        load_frame_interval = self.load_frame_interval
        for sub_frame in sub_frame_list[::load_frame_interval]:
            sub_frame_id = int(sub_frame.split('.')[0])
            one_scan['sub_frames'][sub_frame_id] = {}
            
            if options.get('load_pc_info', True):
                # load pcd data
                pcd_data = np.fromfile(os.path.join(self.embodied_base_dir, self.dataset_name, 'points', scan_id, sub_frame), dtype=np.float32).reshape(-1, 6)
                instance_labels = np.fromfile(os.path.join(self.embodied_base_dir, self.dataset_name, 'instance_mask', scan_id, sub_frame), dtype=np.int64)
                # instance_labels in range 0-max_instance_id, change to -100, 0, 1, 2,...
                instance_labels -= 1
                instance_labels[instance_labels == -1] = -100
                # pre process
                points, colors = pcd_data[:, :3], pcd_data[:, 3:]  
                if self.dataset_name == 'HM3D':
                    points[:, [1, 2]] = points[:, [2, 1]]
                colors = colors / 127.5 - 1
                pcds = np.concatenate([points, colors], 1)
                one_scan['sub_frames'][sub_frame_id]['pcds'] = pcds
                one_scan['sub_frames'][sub_frame_id]['instance_labels'] = instance_labels
                one_scan['sub_frames'][sub_frame_id]['inst_to_label'] = inst_to_label
                    
            if options.get('load_segment_info', False):
                segment_id = np.fromfile(os.path.join(self.embodied_base_dir, self.dataset_name, 'super_points', scan_id, sub_frame), dtype=np.int64)
                unique_ids = np.unique(segment_id)
                assert unique_ids.max() == len(unique_ids) - 1
                one_scan['sub_frames'][sub_frame_id]["segment_id"] = segment_id
            
            if options.get('load_image_segment_feat', False):
                img_feat = np.load(os.path.join(self.embodied_base_dir, self.dataset_name, 'img_feat', scan_id, f'{sub_frame_id}.npy'))
                img_feat = img_feat[unique_ids]
                one_scan['sub_frames'][sub_frame_id]['image_segment_feat'] = img_feat
            
        if options.get('load_global_pc', False):
            # load global pcd
            pcd_data = np.fromfile(os.path.join(self.embodied_base_dir, self.dataset_name, 'points_global', f'{scan_id}.bin'), dtype=np.float32).reshape(-1, 6)
            instance_labels = np.fromfile(os.path.join(self.embodied_base_dir, self.dataset_name, 'instance_mask_global', f'{scan_id}.bin'), dtype=np.int64)
            # instance_labels in range 0-max_instance_id, change to -100, 0, 1, 2,...
            instance_labels -= 1
            instance_labels[instance_labels == -1] = -100
            # pre process
            points, colors = pcd_data[:, :3], pcd_data[:, 3:]
            if self.dataset_name == 'HM3D':
                points[:, [1, 2]] = points[:, [2, 1]]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            one_scan['pcds_global'] = pcds
            one_scan['instance_labels_global'] = instance_labels
                    
        return (scan_id, one_scan)

@DATASET_REGISTRY.register()
class EmbodiedScanInstseg(EmbodiedScanBase):
    def __init__(self, cfg, dataset_name, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, dataset_name, split)
        # load instseg options
        instseg_options = cfg.data.get('instseg_options', {})
        self.ignore_label = instseg_options.get('ignore_label', -100)
        self.filter_out_classes = instseg_options.get('filter_out_classes', [0, 2])
        self.voxel_size = instseg_options.get('voxel_size', 0.02)
        self.use_aug = instseg_options.get('use_aug', True)
        self.num_labels = instseg_options.get('num_labels', 200)
        self.use_open_vocabulary = instseg_options.get('use_open_vocabulary', False)
        # query init params
        self.num_queries = instseg_options.get('num_queries', 120)
        if split != 'train':
            self.query_sample_strategy = 'segment'
        else:
            self.query_sample_strategy = instseg_options.get('query_sample_strategy', 'segment')
        self.compute_local_box = instseg_options.get('compute_local_box', False)
        assert self.query_sample_strategy in ['fps', 'gt', 'segment', 'random_segment']
        # load augmentations
        self.volume_augmentations = V.Compose([])
        self.image_augmentations = A.Compose([])
        if instseg_options.get('volume_augmentations_path', None) is not None:
            self.volume_augmentations = V.load(
                Path(instseg_options.get('volume_augmentations_path', None))
            )
        if instseg_options.get('image_augmentations_path', None) is not None:
            self.image_augmentations = A.load(
                Path(instseg_options.get('image_augmentations_path', None)), data_format="yaml"
            )
        color_mean = [0.47793125906962, 0.4303257521323044, 0.3749598901421883]
        color_std = [0.2834475483823543, 0.27566157565723015, 0.27018971370874995]
        self.normalize_color = A.Normalize(mean=color_mean, std=color_std)
        # init data
        self.scan_ids = self._load_split(self.cfg, self.split)
        self.init_scan_data()
        self.extract_inst_info()
        # build data id mapper, one scene has many sub frame
        self.data_id_mapper = {}
        for scan_id in self.scan_ids:
            sub_frame_ids = sorted(list(self.scan_data[scan_id]['sub_frames'].keys()))
            for sub_frame_id in sub_frame_ids:
                self.data_id_mapper[len(self.data_id_mapper)] = (scan_id, sub_frame_id) 
        
    def __len__(self):
        return len(self.data_id_mapper)
    
    def __getitem__(self, index):
        scan_id, sub_frame_id = self.data_id_mapper[index]
        data_dict = self.get_scene(scan_id, sub_frame_id)
        return data_dict

    def extract_inst_info(self):
        for scan_id in self.scan_ids:
            if self.scan_data[scan_id].get("extract_inst_info", False):
                continue
            for sub_frame_id in self.scan_data[scan_id]['sub_frames'].keys():
                # load useful data
                scan_data = self.scan_data[scan_id]['sub_frames'][sub_frame_id]
                pcds = scan_data['pcds']
                segment_id = scan_data['segment_id']
                instance_labels = scan_data['instance_labels']
                inst_to_label = self.scan_data[scan_id]['inst_to_label']
                inst_to_text_label = self.scan_data[scan_id]['inst_to_text_label']
                # build semantic labels in scannet200
                # 0-199 for ordinary ones, self.ignore_label for undefined semantic and no object
                sem_labels = np.zeros(pcds.shape[0]) + self.ignore_label
                for inst_id in inst_to_label.keys():
                    if inst_to_label[inst_id] in self.cat2int.keys():
                        mask = instance_labels == inst_id
                        if np.sum(mask) == 0:
                            continue
                        sem_labels[mask] = int(self.label_converter.raw_name_to_scannet_raw_id[inst_to_label[inst_id]])
                sem_labels = self.map_to_scannet200_id(sem_labels).astype(int)
                scan_data['sem_labels'] = sem_labels
                # build inst label mapper to map inst label to 0...max
                inst_label_mapper = {}
                max_inst_id = 0
                for inst_id in np.unique(instance_labels):
                    if inst_id in inst_to_label.keys():
                        inst_label_mapper[inst_id] = max_inst_id
                        max_inst_id += 1
                    else:
                        inst_label_mapper[inst_id] = -1 # -1 for no object, so continuous id is -1, 0, 1,... max_obj
                scan_data['inst_label_mapper'] = inst_label_mapper
                instance_labels_continous = np.vectorize(lambda x: inst_label_mapper.get(x, x))(instance_labels)
                scan_data['instance_labels_continuous'] = instance_labels_continous.astype(int)
                # get unique instances, indices same shape with unique inst ids, inverse indices same shape with instance labels
                unique_inst_ids, indices, inverse_indices = np.unique(instance_labels_continous, return_index=True, return_inverse=True)
                unique_inst_labels = sem_labels[indices]
                n_inst = len(unique_inst_ids)
                # instance to point mask
                instance_range = np.arange(n_inst)[:, None]
                full_masks = instance_range == inverse_indices
                # instance to segment mask
                n_segments = segment_id.max() + 1
                segment_masks = np.zeros((n_inst, n_segments), dtype=bool)
                segment_labels = np.ones(n_segments) * self.ignore_label
                for cur_seg_id in range(n_segments):
                    cur_inst_id = np.bincount(inverse_indices[segment_id == cur_seg_id]).argmax()
                    segment_masks[cur_inst_id, cur_seg_id] = True
                    segment_labels[cur_seg_id] = unique_inst_labels[cur_inst_id]
                segment_labels[segment_labels == self.ignore_label] = self.num_labels # set to scannet 200
                # filter out unwanted instances, e.g. wall, floor, and object not in scannet 200
                if self.use_open_vocabulary:
                    valid = (unique_inst_ids != -1) & ~np.isin(unique_inst_labels, self.filter_out_classes) & (segment_masks.sum(1) > 0)
                else:
                    valid = (unique_inst_ids != -1) & ~np.isin(unique_inst_labels, self.filter_out_classes) & (unique_inst_labels != self.ignore_label) & (segment_masks.sum(1) > 0)
                unique_inst_ids = unique_inst_ids[valid]
                unique_inst_labels = unique_inst_labels[valid]
                full_masks = full_masks[valid]
                segment_masks = segment_masks[valid]
                # get text label
                inverse_inst_label_mapper = {v: k for k, v in inst_label_mapper.items()}
                instance_text_labels = [inst_to_text_label[inverse_inst_label_mapper[inst_id]] for inst_id in unique_inst_ids]
                inst_info = {
                    'instance_ids_ori': torch.LongTensor([inverse_inst_label_mapper[inst_id] for inst_id in unique_inst_ids]), # orignal instance id, note for hm3d, instance id - 1
                    'instance_ids': torch.LongTensor(unique_inst_ids), # mapped instance id
                    'instance_labels': torch.LongTensor(unique_inst_labels), # ignore_label, 0-200
                    'full_masks': torch.LongTensor(full_masks), # (n_inst, npoint)
                    'segment_masks': torch.LongTensor(segment_masks), # (n_inst, nsegment_)
                    'segment_labels': torch.LongTensor(segment_labels), # 0-201
                    'instance_text_labels': instance_text_labels # ['class', ...]
                }
                scan_data['inst_info'] = inst_info
            self.scan_data[scan_id]['extract_inst_info'] = True
    
    def sample_query(self, voxel_coordinates, coordinates, obj_center, segment_center):
        if self.query_sample_strategy == 'fps':
            fps_idx = torch.from_numpy(fpsample.bucket_fps_kdline_sampling(voxel_coordinates.numpy(), self.num_queries, h=3).astype(np.int32))
            sampled_coords = coordinates[fps_idx]
            return sampled_coords, torch.ones(len(sampled_coords), dtype=torch.bool), None
        elif self.query_sample_strategy == 'gt':
            return obj_center, torch.ones(len(obj_center), dtype=torch.bool), None
        elif self.query_sample_strategy == 'segment':
            return segment_center, torch.ones(len(segment_center), dtype=torch.bool), torch.arange(len(segment_center))
        elif self.query_sample_strategy == 'random_segment':
            n = 0.5 * torch.rand(1) + 0.5
            n = (n * len(segment_center)).ceil().int()
            assert n > 0
            ids = torch.randperm(len(segment_center))[:n].to(segment_center.device)
            return segment_center[ids], torch.ones(len(ids), dtype=torch.bool), ids
        else:
            raise NotImplementedError(f'{self.query_sample_strategy} is not implemented')

    def get_scene(self, scan_id, sub_frame_id):
        # load local information
        pcds = deepcopy(self.scan_data[scan_id]['sub_frames'][sub_frame_id]['pcds'])
        coordinates = pcds[:, :3]
        color = (pcds[:, :3:] + 1) * 127.5
        point2seg_id = deepcopy(self.scan_data[scan_id]['sub_frames'][sub_frame_id]['segment_id'])
        sem_labels = deepcopy(self.scan_data[scan_id]['sub_frames'][sub_frame_id]['sem_labels']) # self.ignore_label for undefined semantic
        inst_label = deepcopy(self.scan_data[scan_id]['sub_frames'][sub_frame_id]['instance_labels_continuous']) # -1 for no object, so continuous id is -1, 0, 1,... max_obj
        labels = np.concatenate([sem_labels.reshape(-1, 1), inst_label.reshape(-1, 1)], axis=1)
        if self.compute_local_box:
            # load global information
            global_pcds = deepcopy(self.scan_data[scan_id]['pcds_global'])
            global_coordinates = global_pcds[:, :3]
            global_color = (global_pcds[:, :3:] + 1) * 127.5
            global_inst_labels = deepcopy(self.scan_data[scan_id]['instance_labels_global'])
            # concat 
            coordinates = np.concatenate([coordinates, global_coordinates], axis=0)
            color = np.concatenate([color, global_color], axis=0)
         
        if self.split == 'train' and self.use_aug:
            # revser x y axis
            for i in (0, 1):
                if random.random() < 0.5:
                    coord_max = np.max(coordinates[:, i])
                    coord_min = np.min(coordinates[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i] + coord_min
                    
            # volume augmentation
            aug = self.volume_augmentations(
                points=coordinates,
                normals=None,
                features=color,
                labels=None,
            )
            coordinates, color = (
                aug["points"],
                aug["features"],
            )
            # color augmentation
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(
                self.image_augmentations(image=pseudo_image)["image"]
            )
        
        # de concat
        if self.compute_local_box:
            global_coordinates = coordinates[coordinates.shape[0] - global_coordinates.shape[0]:]    
            global_color = color[color.shape[0] - global_color.shape[0]:]
            coordinates = coordinates[:coordinates.shape[0] - global_coordinates.shape[0]]
            color = color[:color.shape[0] - global_color.shape[0]]
        
        # normalize color
        raw_color = color.copy()
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])
        features = np.hstack((color, coordinates))

        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()
        coordinates = torch.from_numpy(coordinates).float()

        # Calculate object centers and segment centers
        instance_info = deepcopy(self.scan_data[scan_id]['sub_frames'][sub_frame_id]['inst_info'])
        instance_ids = instance_info['instance_ids']
        point2inst_id = labels[:, -1]
        point2inst_id[point2inst_id == -1] = (instance_ids.max() if len(instance_ids) else 0) + 1
        obj_center = scatter_mean(coordinates, point2inst_id, dim=0)
        obj_center = obj_center[instance_ids]
        point2seg_id = torch.from_numpy(point2seg_id).long()
        seg_center = scatter_mean(coordinates, point2seg_id, dim=0)
        seg_point_count = scatter_add(torch.ones_like(point2seg_id), point2seg_id, dim=0)

        # voxelize coorinates, features and labels
        voxel_coordinates = np.floor(coordinates / self.voxel_size)
        _, unique_map, inverse_map = ME.utils.sparse_quantize(coordinates=voxel_coordinates, return_index=True, return_inverse=True)
        voxel_coordinates = voxel_coordinates[unique_map]
        voxel_features = features[unique_map]
        voxel2seg_id = point2seg_id[unique_map]
        
        # sample queries
        query_locs, query_pad_masks, query_selection_ids = self.sample_query(voxel_coordinates, coordinates, obj_center, seg_center)
        
        # fill data dict
        data_dict = {
            # input data
            "voxel_coordinates": voxel_coordinates,
            "voxel_features": voxel_features,
            "voxel2segment": voxel2seg_id, # list collate
            "coordinates": voxel_features[:, -3:], # list collate
            # full, voxel, segment mappings
            "voxel_to_full_maps": inverse_map, # list collate
            "segment_to_full_maps": point2seg_id, # list collate
            # for computing iou in raw data
            "raw_coordinates": np.concatenate([coordinates.numpy(), raw_color], axis=1), # only this numpy, list collate
            "coord_min": coordinates.min(0)[0],
            "coord_max": coordinates.max(0)[0],
            "scan_id": scan_id, # str
            "sub_frame_id": str(sub_frame_id), # str
            # extra instance info
            "obj_center": obj_center, 
            "obj_pad_masks": torch.ones(len(obj_center), dtype=torch.bool),
            "seg_center": seg_center,
            "seg_pad_masks": torch.ones(len(seg_center), dtype=torch.bool),
            "seg_point_count": seg_point_count,
            # query info
            'query_locs': query_locs,
            'query_pad_masks': query_pad_masks,
            'query_selection_ids': query_selection_ids
        }
        # instance_info includes instance_ids_ori', 'instance_ids', 'instance_labels', 'full_masks', 'segment_masks', 'segment_labels', 'instance_text_labels' 
        # instance_text_embeds and instance_boxes are optional, torch tensor, list collate
        #  all instance info is list collate
        # instance_text_labels is a list of text
        data_dict.update(instance_info)
        
        if 'image_segment_feat' in self.scan_data[scan_id]['sub_frames'][sub_frame_id].keys():
            cur_segment_image = torch.from_numpy(deepcopy(self.scan_data[scan_id]['sub_frames'][sub_frame_id]['image_segment_feat']))
            data_dict['mv_seg_fts'] = cur_segment_image
            data_dict['mv_seg_pad_masks'] = torch.ones(cur_segment_image.shape[0], dtype=torch.bool)
        
        if self.use_open_vocabulary:
            if self.dataset_name == 'ScanNet':
                instance_text_embeds = [self.scannet_607_cat_to_text_embed[t] for t in data_dict['instance_text_labels']]
            else:
                instance_text_embeds = [self.hm3d_cat_to_text_embed[t] for t in data_dict['instance_text_labels']]
            if len(instance_text_embeds) > 0:
                data_dict['instance_text_embeds'] = torch.stack(instance_text_embeds, dim=0)
            else:
                data_dict['instance_text_embeds'] = torch.empty((0,), dtype=torch.float32)
            
        # compute box
        if self.compute_local_box:
            local_instance_ids = data_dict['instance_ids_ori']
            instance_boxes = []
            for j, instance_id in enumerate(local_instance_ids):
                mask = global_inst_labels == instance_id.item()
                local_pcds = global_coordinates[mask]
                if local_pcds.shape[0] == 0:
                    local_box = convert_pc_to_box(coordinates[data_dict['full_masks'][j].bool()].numpy())
                else:
                    local_box = convert_pc_to_box(local_pcds)
                instance_boxes.append(torch.tensor(local_box[0] + local_box[1], dtype=torch.float32))
            if instance_boxes:
                data_dict['instance_boxes'] = torch.stack(instance_boxes, dim=0)
            else:
                data_dict['instance_boxes'] = torch.empty((0, 6), dtype=torch.float32)

        return data_dict
        
    def map_to_scannet200_id(self, labels):
        label_info = self.label_converter.scannet_raw_id_to_scannet200_id
        labels[~np.isin(labels, list(label_info))] = self.ignore_label
        for k in label_info:
            labels[labels == k] = label_info[k]
        return labels

@DATASET_REGISTRY.register()
class EmbodiedScanInstSegScanNet(EmbodiedScanInstseg):
    def __init__(self, cfg, split):
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        super().__init__(cfg, 'ScanNet', split)


@DATASET_REGISTRY.register()
class EmbodiedScanInstSegHM3D(EmbodiedScanInstseg):
    def __init__(self, cfg, split):
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        super().__init__(cfg, 'HM3D', split)


    
    
    