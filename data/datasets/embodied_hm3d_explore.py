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
from data.datasets.scannet_base import ScanNetBase
from data.datasets.constant import CLASS_LABELS_200, PromptType
from data.data_utils import make_bce_label
import fpsample
from data.datasets.hm3d_label_convert import convert_gpt4

SCAN_DATA = {'HM3D': {}}

class EmbodiedHM3DExploreBase(Dataset, ABC):
    def __init__(self, cfg, dataset_name, split) -> None:
        # basic settings
        self.cfg = cfg
        self.dataset_name = dataset_name # ['ScanNet']
        self.split = split
        self.base_dir = cfg.data.scene_verse_base
        self.embodied_base_dir = cfg.data.embodied_base
        self.load_scan_options = cfg.data.get('load_scan_options', {})
        # label converter
        self.int2cat = json.load(open(os.path.join(self.base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(self.base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2-labels.combined.tsv"))
        self.hm3d_to_scannet607 = convert_gpt4
        hm3d_sem_category_mapping = np.loadtxt(os.path.join(self.embodied_base_dir, "hm3dsem_category_mappings.tsv"), dtype=str, delimiter="\t")
        self.hm3d_raw_to_cat = {hm3d_sem_category_mapping[i, 0]: hm3d_sem_category_mapping[i, 1] for i in range(1, hm3d_sem_category_mapping.shape[0])}
        self.hm3d_cat_to_text_embed = torch.load(os.path.join(self.embodied_base_dir, "hm3d_sem_text_feature.pth"))
    
    def __len__(self):
        return len(self.lang_data)
    
    def __getitem__(self, index):
        scan_id, decision_id, is_object_decision, tgt_object_id_list, data_dict = self.get_lang(index)
        scene_dict = self.get_scene(scan_id, decision_id, is_object_decision, tgt_object_id_list)
        data_dict.update(scene_dict)
        return data_dict
    
    # init datasets
    def init_dataset_params(self, dataset_cfg):
        if dataset_cfg is None:
            dataset_cfg = {}
        self.train_duplicate = dataset_cfg.get('train_duplicate', 1)
        self.frontier_to_class_prob = dataset_cfg.get('frontier_to_class_prob', 0.5)

    def _load_split(self, cfg, split):
        if self.dataset_name == 'HM3D':
            train_val_split = json.load(open(os.path.join(self.embodied_base_dir, 'hm3d_annotated_basis.scene_dataset_config.json')))
            if split == 'train':
                scan_ids = [pa.split('/')[1] for pa in train_val_split['scene_instances']['paths']['.json'] if pa.startswith(split)]
                scan_ids = sorted(scan_ids)
            else:
                scan_ids = [pa.split('/')[1] for pa in train_val_split['scene_instances']['paths']['.json'] if pa.startswith('train')]
                scan_ids = scan_ids[:5]
                scan_ids = sorted(scan_ids)
        else:
            raise NotImplemented(f'data set name {self.dataset_name}')
        # episode not seen and found 
        episode_status = json.load(open(os.path.join(self.embodied_base_dir, 'episode_status.json')))
        wrong_scan_ids = set(episode_status['episodes_not_seen'] + episode_status['episodes_not_found'])
        episode_length = episode_status['episode_decision_length']
        episode_max_invalid_ratio = episode_status['episode_max_invalid_ratio']
        
        all_raw_scan_ids = os.listdir(os.path.join(self.embodied_base_dir, 'points'))
        scan_ids = [raw_scan_id for raw_scan_id in all_raw_scan_ids if raw_scan_id.split('_')[0] in scan_ids]
        scan_ids = [scan_id for scan_id in scan_ids if scan_id not in wrong_scan_ids]
        scan_ids = [scan_id for scan_id in scan_ids if episode_length[scan_id] < 10]
        scan_ids = [scan_id for scan_id in scan_ids if episode_max_invalid_ratio[scan_id] < 50.00]
        
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
        one_scan = {}
        
        # load inst to label
        scan_id_hm3d = scan_id.split('_')[0]
        inst_to_label = torch.load(os.path.join(self.embodied_base_dir, 'instance_id_to_label', f'{scan_id_hm3d}_00.pth')) 
        inst_to_hm3d_label = {k - 1: self.hm3d_raw_to_cat[v] if v in self.hm3d_raw_to_cat.keys() else v for k,v in inst_to_label.items()}
        # change inst to label to scannet category
        inst_to_label = {k - 1: self.hm3d_to_scannet607[v] if v in self.hm3d_to_scannet607.keys() else 'object' for k, v in inst_to_label.items()}
        one_scan['inst_to_label'] = inst_to_label
        one_scan['inst_to_hm3d_label'] = inst_to_hm3d_label
        
        # load inst to box
        instance_id_to_box_path = os.path.join(self.embodied_base_dir, 'instance_id_to_box', scan_id + '.pth')
        instance_id_to_box = torch.load(instance_id_to_box_path)
        one_scan['instance_id_to_box'] = {k - 1: v for k,v in instance_id_to_box.items()}
        for k, v in instance_id_to_box.items():
            v[[1,2]] = v[[2,1]]
            v[[4,5]] = v[[5,4]]
          
        # load decision
        decision_path = os.path.join(self.embodied_base_dir, 'decision', scan_id + '.json')   
        decision_list = []
        decision_info = json.load(open(decision_path, 'r'))
        for cur_decision in decision_info['decision_list']:
            tgt_object_id = int(cur_decision['object_id'].split("_")[1]) - 1
            frontier_list = cur_decision['frontier_list']
            for i in range(len(frontier_list)):
                frontier_list[i] = [frontier_list[i][0], frontier_list[i][2], frontier_list[i][1]]
            decision_list.append({'frontier_list': cur_decision['frontier_list'], 'best_frontier_idx': cur_decision['best_frontier_idx'],
                                  'is_object_decision': cur_decision['is_object_decision'], 'tgt_object_id': tgt_object_id,
                                  'sentece': inst_to_hm3d_label[tgt_object_id]})
        one_scan['decision_list'] = decision_list
        # load query_feat
        query_feat_path = os.path.join(self.embodied_base_dir, 'query_feat', scan_id + '.pth')
        query_feat_list = []
        query_inst_ids_list = []
        query_feat_info = torch.load(query_feat_path)
        for i in range(len(decision_list)):
            query_feat_list.append(query_feat_info[i])
            query_inst_ids_list.append([k for k in query_feat_info[i].keys()])
        one_scan['query_feat_list'] = query_feat_list
        one_scan['query_inst_ids_list'] = query_inst_ids_list
                    
        if options.get('load_global_pc', False):
            # load global pcd
            pcd_data = np.fromfile(os.path.join(self.embodied_base_dir, 'points_global', f'{scan_id}.bin'), dtype=np.float32).reshape(-1, 6)
            instance_labels = np.fromfile(os.path.join(self.embodied_base_dir, 'instance_mask_global', f'{scan_id}.bin'), dtype=np.int64)
            # instance_labels in range 0-max_instance_id, change to -100, 0, 1, 2,...
            instance_labels -= 1
            instance_labels[instance_labels == -1] = -100
            # pre process
            points, colors = pcd_data[:, :3], pcd_data[:, 3:]
            points[:, [1, 2]] = points[:, [2, 1]]  # switch y, z in points
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            one_scan['pcds_global'] = pcds
            one_scan['instance_labels_global'] = instance_labels
                    
        return (scan_id, one_scan)
    
    @abstractmethod
    def get_lang(self, index):
        raise NotImplementedError("This is an abstract class.")
    
    @abstractmethod
    def _load_lang(self):
        raise NotImplementedError("This is an abstract class.")
    
    def get_scene(self, scan_id, decision_id, is_object_decision=True, tgt_object_id_list=[0]):
        # load basic data
        scan_data = deepcopy(self.scan_data[scan_id])
        decision = scan_data['decision_list'][decision_id]
        query_feat = scan_data['query_feat_list'][decision_id]
        inst_to_label = scan_data['inst_to_label']
        inst_to_hm3d_label = scan_data['inst_to_hm3d_label']
        inst_to_box = scan_data['instance_id_to_box']
        frontier_list = decision['frontier_list']
        
        # get tgt_object_id, obj_fts, obj_locs, obj_labels, obj_boxes, obj_pad_masks, real_obj_pad_masks
        obj_ids = [k for k in query_feat.keys()]
        obj_labels = torch.LongTensor([self.cat2int[inst_to_label[obj_id]] for obj_id in obj_ids]).reshape(-1, 1) # N, 1 
        obj_fts = torch.stack([query_feat[obj_id] for obj_id in obj_ids]) # N, 768
        obj_boxes = torch.stack([torch.Tensor(inst_to_box[obj_id]) for obj_id in obj_ids]) # N, 6
        obj_locs = obj_boxes.clone() # N, 6
        obj_pad_masks = torch.ones(len(obj_ids), dtype=torch.bool) # N
        real_obj_pad_masks = torch.ones(len(obj_ids), dtype=torch.bool) # N
        seg_center = obj_locs.clone()
        seg_pad_masks = obj_pad_masks.clone()
        mv_seg_fts = obj_fts.clone()
        mv_seg_pad_masks = obj_pad_masks.clone()
        # add frontier
        num_frontiers = len(frontier_list)
        if num_frontiers > 0:
            obj_ids.extend([-100] * num_frontiers)
            obj_labels = torch.cat((obj_labels, torch.full((num_frontiers, 1), -100)), dim=0)
            obj_fts = torch.cat((obj_fts, torch.zeros(num_frontiers, 768)), dim=0)
            frontier_centers = torch.tensor([frontier[:3] for frontier in frontier_list])
            frontier_boxes = torch.cat((frontier_centers, torch.zeros(num_frontiers, 3)), dim=1)
            obj_boxes = torch.cat((obj_boxes, frontier_boxes), dim=0)
            obj_locs = torch.cat((obj_locs, frontier_boxes), dim=0)
            obj_pad_masks = torch.cat((obj_pad_masks, torch.ones(num_frontiers, dtype=torch.bool)), dim=0)
            real_obj_pad_masks = torch.cat((real_obj_pad_masks, torch.zeros(num_frontiers, dtype=torch.bool)), dim=0)
        if is_object_decision:
            try:
                tgt_object_id = [obj_ids.index(obj_id) for obj_id in tgt_object_id_list]
            except:
                print(f"tgt_object_id {tgt_object_id_list} not in obj_ids, scan_id {scan_id}, decision_id {decision_id}")
                tgt_object_id = []                
                #raise ValueError(f"tgt_object_id {tgt_object_id_list} not in obj_ids {obj_ids}, scan_id {scan_id}, decision_id {decision_id}")
        else:
            tgt_object_id = [real_obj_pad_masks.sum() + obj_id for obj_id in tgt_object_id_list]
        tgt_object_id = torch.LongTensor(tgt_object_id)
        data_dict = {
            "scan_id" : scan_id,
            "decision_id" : decision_id,
            "obj_fts" : obj_fts,
            "obj_locs" : obj_locs,
            "obj_labels" : obj_labels.squeeze(1),
            "obj_boxes" : obj_boxes,
            "obj_pad_masks" : obj_pad_masks,
            "real_obj_pad_masks" : real_obj_pad_masks,
            "tgt_object_id": tgt_object_id,
            "is_object_decision": is_object_decision,
            "seg_center": seg_center,
            "seg_pad_masks": seg_pad_masks,
            "mv_seg_fts": mv_seg_fts,
            "mv_seg_pad_masks": mv_seg_pad_masks
        }
        # preparse query for model
        data_dict['query_locs'] = obj_locs.clone()
        data_dict['query_pad_masks'] = obj_pad_masks.clone()
        return data_dict

@DATASET_REGISTRY.register()
class EmbodiedHM3DExploreMixRefer(EmbodiedHM3DExploreBase):
    def __init__(self, cfg, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'HM3D', split)
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        self.scan_ids = self._load_split(self.cfg, self.split)
        self.init_scan_data()
        self.lang_data = self._load_lang()
    
    def get_lang(self, index):
        item = self.lang_data[index]
        scan_id = item['scan_id']
        decision_id = item['decision_id']
        is_object_decision = item['is_object_decision']
        tgt_object_id_list = item['tgt_object_id_list']
        sentence = item['sentence']
        if not is_object_decision:
            if random.random() < self.frontier_to_class_prob:
                is_object_decision = True
                obj_ids = deepcopy(self.scan_data[scan_id]['query_inst_ids_list'][decision_id])
                inst_id_to_hm3d_label = deepcopy(self.scan_data[scan_id]['inst_to_hm3d_label'])
                obj_id = random.choice(obj_ids)
                sentence = inst_id_to_hm3d_label[obj_id]
                tgt_object_id_list = [i for i in obj_ids if inst_id_to_hm3d_label[i] == sentence]
                
        data_dict = {
            "data_idx": "{scan_id}|{decision_id}".format(scan_id=scan_id, decision_id=decision_id),
            "sentence": sentence,
        }
        return scan_id, decision_id, is_object_decision, tgt_object_id_list, data_dict
    
    def _load_lang(self):
        lang_data = []

        for scan_id in self.scan_ids:
            one_scan = self.scan_data[scan_id]
            for decision_id, decision in enumerate(one_scan['decision_list']):
                if decision['is_object_decision']:
                    lang_data.append({
                        'scan_id': scan_id, 
                        'decision_id': decision_id,
                        'is_object_decision': True,
                        'tgt_object_id_list': [decision['tgt_object_id']],
                        'sentence': decision['sentece']
                    })
                else:
                    lang_data.append({
                        'scan_id': scan_id,
                        'decision_id': decision_id,
                        'is_object_decision': False,
                        'tgt_object_id_list': [decision['best_frontier_idx']],
                        'sentence': decision['sentece']
                    })  
        return lang_data
        

        
        

