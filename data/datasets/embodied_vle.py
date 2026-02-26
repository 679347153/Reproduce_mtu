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
from collections import defaultdict

SCAN_DATA = {'HM3D': {}, 'ScanNet': {}}

"""
init
init_dataset_params
self._load_split
init_scan_data()
"""
class EmbodiedVLEBase(Dataset, ABC):
    def __init__(self, cfg, dataset_name, split) -> None:
        # basic settings
        self.cfg = cfg
        self.dataset_name = dataset_name # ['ScanNet']
        self.split = split
        self.base_dir = cfg.data.scene_verse_base
        self.embodied_base_dir = cfg.data.embodied_base
        self.embodided_feat_dir = cfg.data.embodied_feat
        self.embodied_vle_dir = cfg.data.embodied_vle
        self.load_scan_options = cfg.data.get('load_scan_options', {})
        # label converter
        self.int2cat = json.load(open(os.path.join(self.base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(self.base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2-labels.combined.tsv"))
        self.hm3d_to_scannet607 = convert_gpt4
        hm3d_sem_category_mapping = np.loadtxt(os.path.join(self.embodied_base_dir, 'HM3D', "hm3dsem_category_mappings.tsv"), dtype=str, delimiter="\t")
        self.hm3d_raw_to_cat = {hm3d_sem_category_mapping[i, 0]: hm3d_sem_category_mapping[i, 1] for i in range(1, hm3d_sem_category_mapping.shape[0])}
        self.hm3d_cat_to_text_embed = torch.load(os.path.join(self.embodied_base_dir, 'HM3D', "hm3d_sem_text_feature.pth"))
    
    def __len__(self):
        return len(self.lang_data) * self.train_duplicate
    
    def __getitem__(self, index):
        index = index // self.train_duplicate
        scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict = self.get_lang(index)
        scene_dict = self.get_scene(scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list)
        data_dict.update(scene_dict)
        return data_dict 
    
    def init_dataset_params(self, dataset_cfg):
        if dataset_cfg is None:
            dataset_cfg = {}
        if self.split == 'train':
            self.train_duplicate = dataset_cfg.get('train_duplicate', 1)
        else:
            self.train_duplicate = 1
        if self.split == 'train':
            self.random_drop_ratio = dataset_cfg.get('random_drop_ratio', 0.0)
        else:
            self.random_drop_ratio = 0.0
        self.frontier_to_class_prob = dataset_cfg.get('frontier_to_class_prob', 0.5)
    
    def _load_split(self, cfg, split):
        # load split
        if self.dataset_name == 'HM3D':
            train_val_split = json.load(open(os.path.join(self.embodied_base_dir, 'HM3D', 'hm3d_annotated_basis.scene_dataset_config.json')))
            scan_ids = [pa.split('/')[1] for pa in train_val_split['scene_instances']['paths']['.json'] if pa.startswith(split)] 
            scan_ids = sorted(scan_ids)
        elif self.dataset_name == 'ScanNet':
            if split == 'train':
                split_file = os.path.join(self.base_dir, 'ScanNet/annotations/splits/scannetv2_' + split + "_sort.json")
                with open(split_file, 'r') as f:
                    scan_ids = json.load(f)
            else:
                split_file = os.path.join(self.base_dir, 'ScanNet/annotations/splits/scannetv2_' + split + ".txt")
                scan_ids = {x.strip() for x in open(split_file, 'r', encoding="utf-8")}
                scan_ids = sorted(scan_ids)
        else:
            raise NotImplemented(f'data set name {self.dataset_name}')
        # debug filter
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
        if self.dataset_name == 'HM3D':
            scan_id_hm3d = scan_id
            inst_to_label = torch.load(os.path.join(self.embodied_base_dir, 'HM3D', 'instance_id_to_label', f'{scan_id_hm3d}_00.pth')) 
            inst_to_hm3d_label = {k - 1: self.hm3d_raw_to_cat[v] if v in self.hm3d_raw_to_cat.keys() else v for k,v in inst_to_label.items()}
            # change inst to label to scannet category
            inst_to_label = {k - 1: self.hm3d_to_scannet607[v] if v in self.hm3d_to_scannet607.keys() else 'object' for k, v in inst_to_label.items()}
            one_scan['inst_to_label'] = inst_to_label
            one_scan['inst_to_hm3d_label'] = inst_to_hm3d_label
        elif self.dataset_name == 'ScanNet':
            scan_id_scannet = scan_id
            inst_to_label = torch.load(os.path.join(self.embodied_base_dir, 'ScanNet', 'instance_id_to_label', f'{scan_id_scannet}.pth'))
            one_scan['inst_to_label'] = inst_to_label
        else:
            raise NotImplemented(f'data set name {self.dataset_name}')
        
        # load query feat
        sub_scan_ids =  [sub_scan_id for sub_scan_id in os.listdir(os.path.join(self.embodided_feat_dir, self.dataset_name)) if sub_scan_id.startswith(scan_id)]
        assert len(sub_scan_ids) > 0
        query_feat_dict = defaultdict(lambda : defaultdict(list))
        for sub_scan_id in sub_scan_ids:
            feat_path = os.path.join(self.embodided_feat_dir, self.dataset_name, sub_scan_id)
            feat = torch.load(feat_path)
            for k, v in feat.items(): # k is object id 
                query_feat_dict[k]['object_box'].extend(v['object_box'])
                query_feat_dict[k]['object_feat'].extend(v['object_feat'])
                if options.get('load_score', False):
                    query_feat_dict[k]['object_score'].extend(v['object_score'])
                if options.get('load_openvocab', False):
                    query_feat_dict[k]['object_open_vocab_feat'].extend(v['object_open_vocab_feat'])
        one_scan['query_feat_dict'] = query_feat_dict
        
        return (scan_id, one_scan)
    
    @abstractmethod
    def get_lang(self, index):
        raise NotImplementedError("This is an abstract class.")
    
    @abstractmethod
    def _load_lang(self):
        raise NotImplementedError("This is an abstract class.")
    
    def get_scene(self, scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list):
        # load basic data
        scan_data = deepcopy(self.scan_data[scan_id])
        query_feat_dict = scan_data['query_feat_dict']
        inst_to_label = scan_data['inst_to_label']
         
        # get obj
        if visible_object_id_list is None:
            obj_ids = list(query_feat_dict.keys())
        else:
            obj_ids = visible_object_id_list
        obj_fts = []
        # obj_scores = []
        obj_vocab_fts = []
        obj_boxes = []
        obj_labels = []
        new_obj_ids = []
        for obj_id in obj_ids:
            if obj_id not in query_feat_dict.keys():
                continue
            choice_id = random.randint(0, len(query_feat_dict[obj_id]['object_feat']) - 1)
            obj_feat = query_feat_dict[obj_id]['object_feat'][choice_id]
            obj_openvocab_feat = query_feat_dict[obj_id]['object_open_vocab_feat'][choice_id]
            obj_box = query_feat_dict[obj_id]['object_box'][choice_id]
            # object_score = query_feat_dict[obj_id]['object_score'][choice_id]
            # obj_scores.append(torch.tensor(object_score))
            obj_fts.append(torch.tensor(obj_feat))
            obj_vocab_fts.append(torch.tensor(obj_openvocab_feat))
            obj_boxes.append(torch.tensor(obj_box))
            obj_labels.append(self.cat2int[inst_to_label[obj_id]])
            new_obj_ids.append(obj_id)
        obj_fts = torch.stack(obj_fts, dim=0).float()
        obj_vocab_fts = torch.stack(obj_vocab_fts, dim=0).float()
        # obj_scores = torch.stack(obj_scores, dim=0).float()
        obj_boxes = torch.stack(obj_boxes, dim=0).float()
        obj_labels = torch.LongTensor(obj_labels).reshape(-1, 1) # N, 1
        obj_ids = new_obj_ids
        obj_locs = obj_boxes.clone()
        obj_pad_masks = torch.ones(len(obj_ids), dtype=torch.bool) # N
        real_obj_pad_masks = torch.ones(len(obj_ids), dtype=torch.bool) # N
        # get segment info
        seg_center = obj_locs.clone()
        seg_pad_masks = obj_pad_masks.clone()
        mv_seg_fts = obj_fts.clone()
        mv_seg_pad_masks = obj_pad_masks.clone()
        vocab_seg_fts = obj_vocab_fts.clone()
        vocab_seg_pad_masks = obj_pad_masks.clone()
        
        # add frontier 
        num_frontiers = len(frontier_list)
        if num_frontiers > 0:
            obj_ids.extend([-100] * num_frontiers)
            obj_labels = torch.cat((obj_labels, torch.full((num_frontiers, 1), -100)), dim=0)
            obj_fts = torch.cat((obj_fts, torch.zeros(num_frontiers, 768)), dim=0)
            # obj_scores = torch.cat((obj_scores, torch.ones(num_frontiers)), dim=0)
            frontier_centers = torch.tensor([frontier[:3] for frontier in frontier_list])
            frontier_boxes = torch.cat((frontier_centers, torch.zeros(num_frontiers, 3)), dim=1)
            obj_boxes = torch.cat((obj_boxes, frontier_boxes), dim=0)
            obj_locs = torch.cat((obj_locs, frontier_boxes), dim=0)
            obj_pad_masks = torch.cat((obj_pad_masks, torch.ones(num_frontiers, dtype=torch.bool)), dim=0)
            real_obj_pad_masks = torch.cat((real_obj_pad_masks, torch.zeros(num_frontiers, dtype=torch.bool)), dim=0)
        
        # add target
        if is_object_decision:
            tgt_object_id = [obj_ids.index(obj_id) for obj_id in tgt_object_id_list if obj_id in obj_ids]
        else:
            tgt_object_id = [real_obj_pad_masks.sum() + obj_id for obj_id in tgt_object_id_list] 
        tgt_object_id = torch.LongTensor(tgt_object_id)
        
        # build output dict
        data_dict = {
            "scan_id" : scan_id,
            "decision_id" : decision_id,
            "obj_fts" : obj_fts, # uselesss
            "obj_locs" : obj_locs, # useless
            "obj_labels" : obj_labels.squeeze(1),
            "obj_boxes" : obj_boxes, #useless
            "obj_pad_masks" : obj_pad_masks,
            "real_obj_pad_masks" : real_obj_pad_masks,
            "tgt_object_id": tgt_object_id,
            "is_object_decision": is_object_decision,
            "decision_label": 1 if is_object_decision else 0,
            "seg_center": seg_center,
            "seg_pad_masks": seg_pad_masks,
            "mv_seg_fts": mv_seg_fts,
            "mv_seg_pad_masks": mv_seg_pad_masks,
            "vocab_seg_fts": vocab_seg_fts,
            "vocab_seg_pad_masks": vocab_seg_pad_masks,
        }
        # preparse query for model
        data_dict['query_locs'] = obj_locs.clone()
        data_dict['query_pad_masks'] = obj_pad_masks.clone()
        # data_dict['query_scores'] = obj_scores.clone()
        return data_dict
    
@DATASET_REGISTRY.register() 
class EmbodiedVLEOvon(EmbodiedVLEBase):
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
        episode_id = item['episode_id']
        sub_episode_id = item['sub_episode_id']
        time_step = item['time_step']
        decision_id = f'{episode_id}|{sub_episode_id}|{time_step}'
        visible_object_id_list = item['object_list']
        frontier_list = item['frontier_list']
        is_object_decision = item['is_object_decision']
        tgt_object_id_list = item['tgt_object_id_list']
        sentence = item['sentence']
        if not is_object_decision:
            if self.split == 'train' and random.random() < self.frontier_to_class_prob:
                is_object_decision = True
                obj_id = random.choice(visible_object_id_list)
                sentence = self.scan_data[scan_id]['inst_to_hm3d_label'][obj_id]
                tgt_object_id_list = [i for i in visible_object_id_list if self.scan_data[scan_id]['inst_to_hm3d_label'][i] == sentence]
        data_dict = {
            "data_idx": f'{scan_id}|{decision_id}',
            "sentence": sentence,
        }
        return scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict 
    
    def _load_lang(self):
        lang_data = []
        for scan_id in self.scan_ids:
            if self.split == 'train':
                decision_path = os.path.join(self.embodied_vle_dir, 'ovon', 'train', f'{scan_id}.json')
            else:
                decision_path = os.path.join(self.embodied_vle_dir, 'ovon', 'val_seen', f'{scan_id}.json')
            if not os.path.exists(decision_path):
                continue
            decision_info = json.load(open(decision_path, 'r', encoding="utf-8"))
            visible_ids_scan = set(self.scan_data[scan_id]['query_feat_dict'].keys())
            for episode_id in decision_info.keys():
                episode_data = decision_info[episode_id]
                for sub_episode_id, sub_episode_data in enumerate(episode_data):
                    strategy = sub_episode_data['strategy']
                    status = sub_episode_data['status']
                    decision_list = sub_episode_data['decision_list']
                    for decision in decision_list:
                        cur_dict = {
                            'scan_id': scan_id,
                            'episode_id': episode_id,
                            'sub_episode_id': sub_episode_id,
                            'time_step': decision['time_step'],
                            'object_list': [obj_id - 1 for obj_id in decision['object_list'] if obj_id != 0 and (obj_id - 1) in visible_ids_scan],
                            'is_object_decision': decision['is_object_decision'],
                            'sentence': decision['sentence'],
                        }
                        if len(cur_dict['object_list']) > 300 or len(cur_dict['object_list']) == 0:
                            continue
                        frontier_list = decision['frontier_list']
                        for i in range(len(frontier_list)):
                            frontier_list[i] = [frontier_list[i][0], frontier_list[i][2], frontier_list[i][1]]
                        cur_dict['frontier_list'] = frontier_list
                        if cur_dict['is_object_decision']:
                            cur_dict['tgt_object_id_list'] = [(int(tobj.split('_')[1]) - 1) for tobj in decision['best_object_idx']]
                            cur_dict['tgt_object_id_list'] = [i for i in cur_dict['tgt_object_id_list'] if i in cur_dict['object_list']]
                            if len(cur_dict['tgt_object_id_list']) == 0:
                                continue
                        else:
                            cur_dict['tgt_object_id_list'] = [int(decision['best_frontier_idx'])]
                        lang_data.append(cur_dict)
        if self.random_drop_ratio > 0:
            lang_data = random.sample(lang_data, int(len(lang_data) * (1 - self.random_drop_ratio)))
        return lang_data
    
@DATASET_REGISTRY.register()
class EmbodiedVLEScanRefer(EmbodiedVLEBase):
    def __init__(self, cfg, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'ScanNet', split)
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        self.scan_ids = self._load_split(self.cfg, self.split)
        self.init_scan_data()
        self.lang_data = self._load_lang()

    def get_lang(self, index):
        item = self.lang_data[index]
        scan_id = item['scan_id']
        item_id = item['item_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        decision_id = f"{scan_id}|{item_id}"
        visible_object_id_list = None
        frontier_list = []
        is_object_decision = True
        tgt_object_id_list = [tgt_object_id]

        data_dict = {
            "data_idx": decision_id,
            "sentence": sentence,
        }

        return scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict

    def _load_lang(self):
        split_scan_ids = self.scan_ids
        lang_data = []
        anno_file = os.path.join(self.embodied_vle_dir, 'scannet-refer/scanrefer.jsonl')
        with jsonlines.open(anno_file, 'r') as _f:
            for item in _f:
                if item['scan_id'] in split_scan_ids:
                    lang_data.append(item)
        if self.random_drop_ratio > 0:
            lang_data = random.sample(lang_data, int(len(lang_data) * (1 - self.random_drop_ratio)))
        return lang_data

@DATASET_REGISTRY.register()
class EmbodiedVLEMulti3DRefer(EmbodiedVLEBase):
    def __init__(self, cfg, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'ScanNet', split)
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        self.scan_ids = self._load_split(self.cfg, self.split)
        self.init_scan_data()
        self.lang_data = self._load_lang()

    def get_lang(self, index):
        item = self.lang_data[index]
        scan_id = item['scene_id']
        tgt_object_id = item['object_ids']
        tgt_object_name = [item['object_name'].replace('_', ' ')] * len(tgt_object_id)
        sentence = item['description']
        decision_id = f"{scan_id}|{index}"
        visible_object_id_list = None
        frontier_list = []
        is_object_decision = True
        tgt_object_id_list = tgt_object_id

        data_dict = {
            "data_idx": decision_id,
            "sentence": sentence,
        }

        return scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict

    def _load_lang(self):
        split_scan_ids = self.scan_ids
        lang_data = []
        anno_file = os.path.join(self.embodied_vle_dir, 'scannet-refer', f'multi3drefer_{self.split}.json')
        with open(anno_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if item['scene_id'] in split_scan_ids:
                    lang_data.append(item)
        if self.random_drop_ratio > 0:
            lang_data = random.sample(lang_data, int(len(lang_data) * (1 - self.random_drop_ratio)))
        return lang_data

@DATASET_REGISTRY.register()
class EmbodiedVLEScanQA(EmbodiedVLEBase):
    def __init__(self, cfg, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'ScanNet', split)
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        self.scan_ids = self._load_split(self.cfg, self.split)
        self.init_scan_data()
        self.lang_data = self._load_lang()

    def get_lang(self, index):
        item = self.lang_data[index]
        scan_id = item['scene_id']
        question_id = item['question_id']
        question = item['question']
        tgt_object_id = [] if self.split == 'test' else item['object_ids']
        tgt_object_name = [] if self.split == 'test' else item['object_names']
        sentence = question
        decision_id = f"{scan_id}|{question_id}"
        visible_object_id_list = None
        frontier_list = []
        is_object_decision = True
        tgt_object_id_list = tgt_object_id

        data_dict = {
            "data_idx": decision_id,
            "sentence": sentence,
        }

        return scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict

    def _load_lang(self):
        lang_data = []
        split_scan_ids= self.scan_ids
        if self.split == 'test':
            json_data = []
            for type in ['w_obj', 'wo_obj']:
                anno_file = os.path.join(self.embodied_vle_dir, 'scannet-refer', f'ScanQA_v1.0_{self.split}_{type}.json')
                json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
        else:
            anno_file = os.path.join(self.embodied_vle_dir, 'scannet-refer', f'ScanQA_v1.0_{self.split}.json')
            json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
            
        for item in json_data:
            if item['scene_id'] in split_scan_ids:
                lang_data.append(item)
        
        if self.random_drop_ratio > 0:
            lang_data = random.sample(lang_data, int(len(lang_data) * (1 - self.random_drop_ratio)))
        return lang_data

@DATASET_REGISTRY.register()
class EmbodiedVLENr3D(EmbodiedVLEBase):
    def __init__(self, cfg, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'ScanNet', split)
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        self.scan_ids = self._load_split(self.cfg, self.split)
        self.init_scan_data()
        self.lang_data = self._load_lang()

    def get_lang(self, index):
        item = self.lang_data[index]
        scan_id = item['scan_id']
        item_id = item['item_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        decision_id = f"{scan_id}|{item_id}"
        visible_object_id_list = None
        frontier_list = []
        is_object_decision = True
        tgt_object_id_list = [tgt_object_id]

        data_dict = {
            "data_idx": decision_id,
            "sentence": sentence,
        }

        return scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict

    def _load_lang(self):
        split_scan_ids = self.scan_ids
        lang_data = []
        anno_file = os.path.join(self.embodied_vle_dir, 'scannet-refer/nr3d.jsonl')
        with jsonlines.open(anno_file, 'r') as _f:
            for item in _f:
                if item['scan_id'] in split_scan_ids:
                    lang_data.append(item)
        if self.random_drop_ratio > 0:
            lang_data = random.sample(lang_data, int(len(lang_data) * (1 - self.random_drop_ratio)))
        return lang_data

@DATASET_REGISTRY.register()
class EmbodiedVLESG3DReferScanNet(EmbodiedVLEBase):
    def __init__(self, cfg, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'ScanNet', split)
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        self.scan_ids = self._load_split(self.cfg, self.split)
        self.init_scan_data()
        self.lang_data = self._load_lang()

    def get_lang(self, index):
        item = self.lang_data[index]
        scan_id = item['scan_id']
        item_id = item['item_id']
        tgt_object_id = [int(action['target_id']) for action in item['action_steps']]
        sentence = item['task_description'] + ' ' + ' '.join([action['action'] for action in item['action_steps']])
        decision_id = f"{scan_id}|{item_id}"
        visible_object_id_list = None
        frontier_list = []
        is_object_decision = True
        tgt_object_id_list = tgt_object_id

        data_dict = {
            "data_idx": decision_id,
            "sentence": sentence,
        }

        return scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict

    def _load_lang(self):
        lang_data = []
        split_scan_ids = self.scan_ids
        # filter data to current dataset
        if self.split == 'train':
            anno_file_base = os.path.join(self.embodied_vle_dir, 'sg3d', 'train.json')
        else:
            anno_file_base = os.path.join(self.embodied_vle_dir, 'sg3d', 'val.json')
        anno_data_pre_filter = json.load(open(anno_file_base))
        anno_data = []
        for data in anno_data_pre_filter:
            dataset_name = data['scan_id'].split('_')[0]
            if dataset_name == self.dataset_name:
                anno_data.append(data)
        # read data
        for i, data in enumerate(anno_data):
            dataset_name = data['scan_id'].split('_')[0]
            scan_id = data['scan_id'][len(self.dataset_name) + 1:]
            if scan_id in split_scan_ids:
                item = {'task_description': data['task_description'], 'action_steps': data['action_steps']}
                item['scan_id'] = scan_id
                item['item_id'] = f'f{scan_id}_{i}'
                for j in range(len(item['action_steps'])):
                    new_item = deepcopy(item)
                    new_action_sentence = " ".join([action['action'] for action in item['action_steps'][:j + 1]])
                    new_item['action_steps'] = [new_item['action_steps'][j]]
                    new_item['action_steps'][0]['action'] = new_action_sentence
                    lang_data.append(new_item) 
        if self.random_drop_ratio > 0:
            lang_data = random.sample(lang_data, int(len(lang_data) * (1 - self.random_drop_ratio)))
        return lang_data 

@DATASET_REGISTRY.register()
class EmbodiedVLESG3DReferHM3D(EmbodiedVLEBase):
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
        scan_id = item['scan_id'] # is sub room id
        raw_scan_id = scan_id.split('_')[0]
        item_id = item['item_id']
        tgt_object_id = [int(action['target_id']) for action in item['action_steps']]
        sentence = item['task_description'] + ' ' + ' '.join([action['action'] for action in item['action_steps']])
        decision_id = f"{scan_id}|{item_id}"
        visible_object_id_list = self.scan_id_to_visible_ids[scan_id]
        frontier_list = []
        is_object_decision = True
        tgt_object_id_list = tgt_object_id

        data_dict = {
            "data_idx": decision_id,
            "sentence": sentence,
        }

        return raw_scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict

    def _load_lang(self):
        lang_data = []
        split_scan_ids = self.scan_ids
        # filter data to current dataset
        if self.split == 'train':
            anno_file_base = os.path.join(self.embodied_vle_dir, 'sg3d', 'train.json')
        else:
            anno_file_base = os.path.join(self.embodied_vle_dir, 'sg3d', 'val.json')
        anno_data_pre_filter = json.load(open(anno_file_base))
        anno_data = []
        for data in anno_data_pre_filter:
            dataset_name = data['scan_id'].split('_')[0]
            if dataset_name == self.dataset_name:
                anno_data.append(data)
        # load id converter
        id_convert_base_path = os.path.join(self.embodied_vle_dir, 'hm3d-refer', 'sceneverse-to-hm3d-raw-id-convert')
        hm3d_scenevserse_id_to_raw_id = defaultdict(dict) # this is raw scan id
        for scan_id in split_scan_ids:
            id_convert_file = os.path.join(id_convert_base_path, f'{scan_id}.json')
            id_convert = json.load(open(id_convert_file))
            for k in id_convert.keys():
                hm3d_scenevserse_id_to_raw_id[scan_id][int(k)] = int(id_convert[k][0])
        # load visible ids
        scan_id_to_visible_ids = {} # this is subscan id
        instance_id_per_scene = json.load(open(os.path.join(self.embodied_vle_dir, 'hm3d-refer', 'instance_id_per_scene.json'), 'r', encoding="utf-8"))
        for scan_id in instance_id_per_scene.keys():
            raw_scan_id = scan_id.split('_')[0]
            if raw_scan_id in split_scan_ids:
                visible_ids_scan = set(self.scan_data[raw_scan_id]['query_feat_dict'].keys())
                visible_ids = [hm3d_scenevserse_id_to_raw_id[raw_scan_id][int(k)] - 1 for k in instance_id_per_scene[scan_id]]
                visible_ids = [k for k in visible_ids if k in visible_ids_scan]
                scan_id_to_visible_ids[scan_id] = visible_ids
        self.scan_id_to_visible_ids = scan_id_to_visible_ids
        # read data
        for i, data in enumerate(anno_data):
            dataset_name = data['scan_id'].split('_')[0]
            scan_id = data['scan_id'][len(self.dataset_name) + 1:]
            raw_scan_id = scan_id.split('_')[0]
            if raw_scan_id in split_scan_ids and len(self.scan_id_to_visible_ids[scan_id]) > 5:
                item = {'task_description': data['task_description'], 'action_steps': data['action_steps']}
                item['scan_id'] = scan_id
                item['item_id'] = f'f{scan_id}_{i}'
                # convert id
                for action in item['action_steps']:
                    action['target_id'] = hm3d_scenevserse_id_to_raw_id[raw_scan_id][int(action['target_id'])] - 1
                # split actions
                for j in range(len(item['action_steps'])):
                    new_item = deepcopy(item)
                    new_action_sentence = " ".join([action['action'] for action in item['action_steps'][:j + 1]])
                    new_item['action_steps'] = [new_item['action_steps'][j]]
                    new_item['action_steps'][0]['action'] = new_action_sentence
                    if new_item['action_steps'][0]['target_id'] in self.scan_id_to_visible_ids[scan_id]:
                        lang_data.append(new_item)
        if self.random_drop_ratio > 0:
            lang_data = random.sample(lang_data, int(len(lang_data) * (1 - self.random_drop_ratio)))
        return lang_data

@DATASET_REGISTRY.register()
class EmbodiedVLEHM3DRefer(EmbodiedVLEBase):
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
        scan_id = item['scan_id'] # is sub room id
        raw_scan_id = scan_id.split('_')[0]
        item_id = item['item_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        decision_id = f"{scan_id}|{item_id}"
        visible_object_id_list = self.scan_id_to_visible_ids[scan_id]
        frontier_list = []
        is_object_decision = True
        tgt_object_id_list = [tgt_object_id]

        data_dict = {
            "data_idx": decision_id,
            "sentence": sentence,
        }

        return raw_scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict

    def _load_lang(self):
        split_scan_ids = self.scan_ids
        lang_data = []
        # anno_file_list = ['anno.json', 'ssg_ref_chain_gpt.json']
        anno_file_list = ['anno.json']
        anno_file_list = [os.path.join(self.embodied_vle_dir, 'hm3d-refer', f'{anno_file}') for anno_file in anno_file_list]
        # load id converter, raw scan id
        id_convert_base_path = os.path.join(self.embodied_vle_dir, 'hm3d-refer', 'sceneverse-to-hm3d-raw-id-convert') 
        hm3d_scenevserse_id_to_raw_id = defaultdict(dict)
        for scan_id in split_scan_ids:
            id_convert_file = os.path.join(id_convert_base_path, f'{scan_id}.json')
            id_convert = json.load(open(id_convert_file))
            for k in id_convert.keys():
                hm3d_scenevserse_id_to_raw_id[scan_id][int(k)] = int(id_convert[k][0])
        # load visible ids, sub room scan id
        scan_id_to_visible_ids = {} 
        instance_id_per_scene = json.load(open(os.path.join(self.embodied_vle_dir, 'hm3d-refer', 'instance_id_per_scene.json'), 'r', encoding="utf-8"))
        for scan_id in instance_id_per_scene.keys():
            raw_scan_id = scan_id.split('_')[0]
            if raw_scan_id in split_scan_ids:
                visible_ids_scan = set(self.scan_data[raw_scan_id]['query_feat_dict'].keys())
                visible_ids = [hm3d_scenevserse_id_to_raw_id[raw_scan_id][int(k)] - 1 for k in instance_id_per_scene[scan_id]]
                visible_ids = [k for k in visible_ids if k in visible_ids_scan]
                scan_id_to_visible_ids[scan_id] = visible_ids
        self.scan_id_to_visible_ids = scan_id_to_visible_ids # scan id is sub room scan id
        # load language
        for anno_file in anno_file_list:
            with open(anno_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    raw_scan_id = item['scan_id'].split('_')[0]
                    if raw_scan_id in split_scan_ids and len(self.scan_id_to_visible_ids[item['scan_id']]) > 5:
                        item['target_id'] = hm3d_scenevserse_id_to_raw_id[raw_scan_id][int(item['target_id'])] - 1
                        if item['target_id'] in self.scan_id_to_visible_ids[item['scan_id']]:
                            lang_data.append(item)
        if self.random_drop_ratio > 0:
            lang_data = random.sample(lang_data, int(len(lang_data) * (1 - self.random_drop_ratio)))
        return lang_data

@DATASET_REGISTRY.register()
class EmbodiedVLESG3D(EmbodiedVLEBase):
    def __init__(self, cfg, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'HM3D', split)
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        self.scan_ids = self._load_split(self.cfg, self.split)
        self.init_scan_data()
        self.lang_data = self._load_lang()
        self.use_single_step = dataset_cfg.get('use_single_step', False)

    def get_lang(self, index):
        item = self.lang_data[index]
        scan_id = item['scan_id']
        episode_id = item['episode_id']
        sub_episode_id = item['sub_episode_id']
        time_step = item['time_step']
        decision_id = f'{episode_id}|{sub_episode_id}|{time_step}'
        visible_object_id_list = item['object_list']
        frontier_list = item['frontier_list']
        is_object_decision = item['is_object_decision']
        tgt_object_id_list = item['tgt_object_id_list']
        sentence = item['sentence']
        if not is_object_decision:
            if self.split == 'train' and random.random() < self.frontier_to_class_prob:
                is_object_decision = True
                obj_id = random.choice(visible_object_id_list)
                sentence = self.scan_data[scan_id]['inst_to_hm3d_label'][obj_id]
                tgt_object_id_list = [i for i in visible_object_id_list if self.scan_data[scan_id]['inst_to_hm3d_label'][i] == sentence]
        if self.use_single_step:
            sentence = sentence.split('. ')[-1]
        data_dict = {
            "data_idx": f'{scan_id}|{decision_id}',
            "sentence": sentence,
        }
        return scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict

    def _load_lang(self):
        lang_data = []
        for scan_id in self.scan_ids:
            if self.split == 'train':
                decision_path = os.path.join(self.embodied_vle_dir, 'sg3d', 'train', f'{scan_id}.json')
            else:
                decision_path = os.path.join(self.embodied_vle_dir, 'sg3d', 'val', f'{scan_id}.json')
            if not os.path.exists(decision_path):
                continue
            decision_info = json.load(open(decision_path, 'r', encoding="utf-8"))
            visible_ids_scan = set(self.scan_data[scan_id]['query_feat_dict'].keys())
            for episode_id in decision_info.keys():
                episode_data = decision_info[episode_id]
                for sub_episode_id, sub_episode_data in enumerate(episode_data):
                    strategy = sub_episode_data['strategy']
                    status = sub_episode_data['status']
                    decision_list = sub_episode_data['decision_list']
                    for decision in decision_list:
                        cur_dict = {
                            'scan_id': scan_id,
                            'episode_id': episode_id,
                            'sub_episode_id': sub_episode_id,
                            'time_step': decision['time_step'],
                            'object_list': [obj_id - 1 for obj_id in decision['object_list'] if obj_id != 0 and (obj_id - 1) in visible_ids_scan],
                            'is_object_decision': decision['is_object_decision'],
                            'sentence': decision['sentence'],
                        }
                        if len(cur_dict['object_list']) > 300 or len(cur_dict['object_list']) == 0:
                            continue
                        frontier_list = decision['frontier_list']
                        for i in range(len(frontier_list)):
                            frontier_list[i] = [frontier_list[i][0], frontier_list[i][2], frontier_list[i][1]]
                        cur_dict['frontier_list'] = frontier_list
                        if cur_dict['is_object_decision']:
                            cur_dict['tgt_object_id_list'] = [(int(tobj.split('_')[1]) - 1) for tobj in decision['best_object_idx']]
                            cur_dict['tgt_object_id_list'] = [i for i in cur_dict['tgt_object_id_list'] if i in cur_dict['object_list']]
                            if len(cur_dict['tgt_object_id_list']) == 0:
                                continue
                        else:
                            cur_dict['tgt_object_id_list'] = [int(decision['best_frontier_idx'])]
                        lang_data.append(cur_dict)
        if self.random_drop_ratio > 0:
            lang_data = random.sample(lang_data, int(len(lang_data) * (1 - self.random_drop_ratio)))
        return lang_data

@DATASET_REGISTRY.register()
class EmbodiedVLEGoat(EmbodiedVLEBase):
    def __init__(self, cfg, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'HM3D', split)
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        self.scan_ids = self._load_split(self.cfg, self.split)
        self.init_scan_data()
        self.lang_data = self._load_lang()
        # load image feature
        image_feat_dir = os.path.join(self.embodied_vle_dir, 'goat-clip-feat')
        image_feat_dict = {}
        for scan_id in self.scan_ids:
            if self.split == 'train':
                if os.path.exists(os.path.join(image_feat_dir, 'train', f'{scan_id}.pt')):
                    image_feat = torch.load(os.path.join(image_feat_dir, 'train', f'{scan_id}.pt'), map_location='cpu')
            else:
                if os.path.exists(os.path.join(image_feat_dir, 'val_seen', f'{scan_id}.pt')):
                    image_feat = torch.load(os.path.join(image_feat_dir, 'val_seen', f'{scan_id}.pt'), map_location='cpu')
            image_feat_dict[scan_id] = image_feat
        self.image_feat_dict = image_feat_dict

    def get_lang(self, index):
        item = self.lang_data[index]
        scan_id = item['scan_id']
        episode_id = item['episode_id']
        sub_episode_id = item['sub_episode_id']
        time_step = item['time_step']
        decision_id = f'{episode_id}|{sub_episode_id}|{time_step}'
        visible_object_id_list = item['object_list']
        frontier_list = item['frontier_list']
        is_object_decision = item['is_object_decision']
        tgt_object_id_list = item['tgt_object_id_list']
        sentence = item['sentence']
        if not is_object_decision and item['task_type'] != 'image':
            if self.split == 'train' and random.random() < self.frontier_to_class_prob:
                is_object_decision = True
                obj_id = random.choice(visible_object_id_list)
                sentence = self.scan_data[scan_id]['inst_to_hm3d_label'][obj_id]
                tgt_object_id_list = [i for i in visible_object_id_list if self.scan_data[scan_id]['inst_to_hm3d_label'][i] == sentence]
        data_dict = {
            "data_idx": f'{scan_id}|{decision_id}',
            "sentence": sentence,
            'is_image_prompt': item['task_type'] == 'image',
        }
        return scan_id, decision_id, is_object_decision, tgt_object_id_list, visible_object_id_list, frontier_list, data_dict

    def _load_lang(self):
        lang_data = []
        for scan_id in self.scan_ids:
            if self.split == 'train':
                decision_path = os.path.join(self.embodied_vle_dir, 'goat-bench', 'train', f'{scan_id}.json')
            else:
                decision_path = os.path.join(self.embodied_vle_dir, 'goat-bench', 'val_seen', f'{scan_id}.json')
            if not os.path.exists(decision_path):
                continue
            decision_info = json.load(open(decision_path, 'r', encoding="utf-8"))
            visible_ids_scan = set(self.scan_data[scan_id]['query_feat_dict'].keys())
            for episode_id in decision_info.keys():
                episode_data = decision_info[episode_id]
                for sub_episode_id, sub_episode_data in enumerate(episode_data):
                    strategy = sub_episode_data['strategy']
                    status = sub_episode_data['status']
                    task_type = sub_episode_data['task_type']
                    # currently we skip image
                    decision_list = sub_episode_data['decision_list']
                    for decision in decision_list:
                        cur_dict = {
                            'scan_id': scan_id,
                            'episode_id': episode_id,
                            'sub_episode_id': sub_episode_id,
                            'time_step': decision['time_step'],
                            'object_list': [obj_id - 1 for obj_id in decision['object_list'] if obj_id != 0 and (obj_id - 1) in visible_ids_scan],
                            'is_object_decision': decision['is_object_decision'],
                            'sentence': decision['sentence'],
                            'task_type': task_type,
                        }
                        if len(cur_dict['object_list']) > 300 or len(cur_dict['object_list']) == 0:
                            continue
                        frontier_list = decision['frontier_list']
                        for i in range(len(frontier_list)):
                            frontier_list[i] = [frontier_list[i][0], frontier_list[i][2], frontier_list[i][1]]
                        cur_dict['frontier_list'] = frontier_list
                        if cur_dict['is_object_decision']:
                            cur_dict['tgt_object_id_list'] = [(int(tobj.split('_')[1]) - 1) for tobj in decision['best_object_idx']]
                            cur_dict['tgt_object_id_list'] = [i for i in cur_dict['tgt_object_id_list'] if i in cur_dict['object_list']]
                            if len(cur_dict['tgt_object_id_list']) == 0:
                                continue
                        else:
                            cur_dict['tgt_object_id_list'] = [int(decision['best_frontier_idx'])]
                        lang_data.append(cur_dict)
        if self.random_drop_ratio > 0:
            lang_data = random.sample(lang_data, int(len(lang_data) * (1 - self.random_drop_ratio)))
        return lang_data
