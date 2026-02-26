import random
import torch
from transformers import AutoTokenizer

from data.datasets.constant import PromptType

from ..data_utils import make_bce_label, pad_sequence_2d, pad_sequence
from .dataset_wrapper import DATASETWRAPPER_REGISTRY
from torch.utils.data import Dataset, default_collate
from .scanfamily_wrapper import ScanFamilyDatasetWrapper

# 0 for refer task, 1 for qa task, 2 for caption task
dataset2task_id = {'EmbodiedVLEOvon': 0, 'EmbodiedVLEScanRefer': 0, 'EmbodiedVLEMulti3DRefer': 0, 'EmbodiedVLEScanQA': 0, 'EmbodiedVLENr3D': 0, 
                   'EmbodiedVLESG3DReferScanNet': 0, 'EmbodiedVLESG3DReferHM3D': 0, 'EmbodiedVLEHM3DRefer': 0, 'EmbodiedVLESG3D': 0, 'EmbodiedVLEGoat': 0}

@DATASETWRAPPER_REGISTRY.register()
class EmbodiedVLEWrapper(Dataset):
    def __init__(self, cfg, dataset):
        self.dataset = dataset
        self.dataset_name = dataset.__class__.__name__
        tokenizer_name = getattr(cfg.data_wrapper, 'tokenizer', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.task_id = dataset2task_id[self.dataset_name]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        data_dict['tgt_object_id'] = make_bce_label(data_dict['tgt_object_id'], num_classes=len(data_dict['obj_labels']))
        if 'is_image_prompt' in data_dict.keys() and data_dict['is_image_prompt']:
            sentence = data_dict['sentence']
            obj_id, image_id = sentence.split('_')
            data_dict['prompt'] = self.dataset.image_feat_dict[data_dict['scan_id']][int(obj_id)][int(image_id)].clone() # in this dict we use hm3d original id!!!!
            data_dict['prompt_pad_masks'] = torch.ones((1)).bool()
            data_dict['prompt_type'] = PromptType.IMAGE
        else:
            data_dict['prompt_type'] = PromptType.TXT
        if 'is_image_prompt' in data_dict.keys():
            del data_dict['is_image_prompt']
        return data_dict
             
    def collate_fn(self, batch):
        new_batch = {}
        # tokenize sentence
        sentences = [data_dict['sentence'] for data_dict in batch]
        encoded_input = self.tokenizer(sentences, add_special_tokens=True, truncation=True)
        tokenized_txt = encoded_input.input_ids
        for i, txt in enumerate(tokenized_txt):
            if batch[i]['prompt_type'] == PromptType.TXT:
                batch[i]['prompt'] = torch.FloatTensor(txt)
                batch[i]['prompt_pad_masks'] = torch.ones((len(txt))).bool()
        # collate list keys
        list_keys = ['prompt']
        list_keys = [k for k in list_keys if k in batch[0].keys()]
        for k in list_keys:
            new_batch[k] = [sample.pop(k) for sample in batch]
        # collate padding keys
        padding_keys = ["obj_fts", "obj_locs", "obj_labels", "obj_boxes", "obj_pad_masks", "real_obj_pad_masks", "tgt_object_id", "seg_center", "seg_pad_masks", "mv_seg_fts", "mv_seg_pad_masks", 'vocab_seg_fts', 'vocab_seg_pad_masks', 'prompt_pad_masks', 'query_locs', 'query_scores', 'query_pad_masks']
        padding_keys = [k for k in padding_keys if k in batch[0].keys()]
        for k in padding_keys:
            tensors = [sample.pop(k) for sample in batch]
            padding_value = -100 if k == 'obj_labels' else 0
            padded_tensor = pad_sequence(tensors, pad=padding_value)
            new_batch[k] = padded_tensor
        # collate others
        new_batch.update(default_collate(batch))
        
        return new_batch
            
        
        
            
    