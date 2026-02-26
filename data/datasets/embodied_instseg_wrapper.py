"""
此文件定义了用于具身化实例分割任务的PyTorch数据集包装器。

文件架构:
1.  `EmbodiedInstSegDatasetWrapper`: 一个标准的数据集包装器，它接收一个基础数据集，并提供一个自定义的 `collate_fn` 方法。
    这个 collate 函数负责将批次中的单个样本打包成一个适用于模型输入的批次张量。它专门处理稀疏体素数据（使用MinkowskiEngine）、
    需要填充的变长张量以及其他类型的数据。

2.  `EmbodiedRecurrentInstSegDatasetWrapper`: 一个用于循环或序列化数据的变体。它处理的是一个批次的样本列表（例如，代表时间序列或导航步骤），
    并对序列中的每个时间步应用与 `EmbodiedInstSegDatasetWrapper` 类似的 collate 逻辑，最终生成一个批次字典的列表。

实现功能:
-   **数据打包 (Collation)**: 核心功能是将来自数据集的样本列表转换成一个统一的批次，以便进行模型训练或推理。
-   **稀疏数据处理**: 使用 `MinkowskiEngine.utils.sparse_collate` 高效地处理和打包稀疏的3D体素坐标和特征。
-   **变长数据填充**: 对批次中长度不一的数据（如分割掩码、边界框等）进行填充，使其具有相同的维度。
-   **目标生成**: 根据离线掩码源（如 'gt'），为损失计算生成目标标签和掩码。
-   **支持循环数据**: 为需要按时间步处理的循环模型（如RNN）提供专门的数据打包逻辑。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from transformers import AutoTokenizer
import MinkowskiEngine as ME

from data.datasets.constant import PromptType

from .dataset_wrapper import DATASETWRAPPER_REGISTRY
from ..data_utils import pad_sequence, pad_sequence_2d


@DATASETWRAPPER_REGISTRY.register()
class EmbodiedInstSegDatasetWrapper(Dataset):
    """
    一个标准的数据集包装器，用于处理具身化实例分割任务的数据。
    它包装了一个基础数据集，并提供一个自定义的 collate_fn 来将数据样本批处理成模型输入格式。
    """
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.offline_mask_source = self.dataset.offline_mask_source
        
    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """获取指定索引的数据样本。"""
        return self.dataset[idx]
    
    def collate_fn(self, batch):
        """
        自定义的collate函数，用于将一批样本打包成一个字典。
        Args:
            batch (list): 一个包含多个数据样本字典的列表。
        Returns:
            dict: 一个包含批处理后张量的字典，可直接用于模型输入。
        """
        new_batch = {}

        # 1. 使用MinkowskiEngine对稀疏体素特征进行批处理
        input_dict = {
            "coords": [sample.pop('voxel_coordinates') for sample in batch], 
            "feats": [sample.pop('voxel_features') for sample in batch],
        }
        # sparse_collate将坐标和特征列表转换成单个批处理后的稀疏张量
        voxel_coordinates, voxel_features = ME.utils.sparse_collate(**input_dict, dtype=torch.int32)
        new_batch['voxel_coordinates'] = voxel_coordinates
        new_batch['voxel_features'] = voxel_features
        
        # 2. 加载和处理离线掩码（如果提供）
        if self.offline_mask_source is not None:
            # 如果掩码源是 'gt' (ground truth)
            if self.offline_mask_source == 'gt':
                # a. 从GT分割掩码构建注意力掩码 (True表示被遮蔽)
                # 对2D分割掩码进行填充，使其在批次中具有相同的尺寸
                padded_segment_masks, padding_mask = pad_sequence_2d([sample['segment_masks'] for sample in batch], return_mask=True)
                # logical_not() 将掩码反转，用于注意力机制 (True的位置是未被mask的)
                new_batch['offline_attn_mask'] = padded_segment_masks.logical_not()
                
                # b. 为损失计算构建标签和掩码
                # 填充实例标签，-100通常用于在损失计算中忽略
                labels = pad_sequence([sample['instance_labels'] for sample in batch], pad=-100)
                new_batch['target_labels'] = labels
                new_batch['target_masks'] = padded_segment_masks.float()
                new_batch['target_masks_pad_masks'] = padding_mask.logical_not()
            else: 
                raise NotImplementedError(f'{self.offline_mask_source} is not implemented')
            
        # 3. 列表批处理：将批次中每个样本的某些键值收集到一个列表中
        # 这些数据通常是变长的，或者不适合堆叠成单个张量
        list_keys = ['voxel2segment', 'coordinates', 'voxel_to_full_maps', 'segment_to_full_maps', 'raw_coordinates', 'instance_ids', 'instance_labels', 'instance_boxes', 'instance_ids_ori', 'full_masks', 'segment_masks', 'scan_id', 'segment_labels', 'query_selection_ids', 'instance_hm3d_labels', 'instance_hm3d_text_embeds']
        list_keys = [k for k in list_keys if k in batch[0].keys()]
        for k in list_keys:
            new_batch[k] = [sample.pop(k) for sample in batch]

        # 4. 填充批处理：对需要填充以形成单个张量的键进行处理
        padding_keys = ['coord_min', 'coord_max', 'obj_center', 'obj_pad_masks', 'seg_center', 'seg_pad_masks', 'seg_point_count', 'query_locs', 'query_pad_masks',
                        'voxel_seg_pad_masks', 'mv_seg_fts', 'mv_seg_pad_masks', 'pc_seg_fts', 'pc_seg_pad_masks', 'prompt', 'prompt_pad_masks']
        padding_keys =  [k for k in padding_keys if k in batch[0].keys()]
        for k in padding_keys:
            tensors = [sample.pop(k) for sample in batch]
            padded_tensor = pad_sequence(tensors)
            new_batch[k] = padded_tensor
        
        # 5. 默认批处理：对其余的键使用PyTorch的默认collate函数
        new_batch.update(default_collate(batch))
        return new_batch

@DATASETWRAPPER_REGISTRY.register()
class EmbodiedRecurrentInstSegDatasetWrapper(Dataset):
    """
    用于处理循环/序列化具身实例分割任务的数据集包装器。
    它处理的是一个批次的样本列表（例如，代表时间序列），并对序列中的每个时间步应用collate逻辑。
    """
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.offline_mask_source = self.dataset.offline_mask_source
        
    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """获取指定索引的数据样本序列。"""
        return self.dataset[idx]
    
    def collate_fn(self, batch_list):
        """
        为循环数据定制的collate函数。
        它迭代处理一个批次的样本列表，其中每个子列表代表一个时间步。
        Args:
            batch_list (list of lists): e.g., [[dict0_t0, dict0_t1], [dict1_t0, dict1_t1], ...]
                                        每个内部列表是一个样本的时间序列。
        Returns:
            dict: 包含一个批处理后的字典列表 `batch_list`，以及帧信息。
        """
        # batch_list 的结构: [[sample0_frame0, sample0_frame1], [sample1_frame0, sample1_frame1], ...]
        # 我们需要将它转换成: [[sample0_frame0, sample1_frame0], [sample0_frame1, sample1_frame1], ...]
        # 然后对每个帧的批次进行collate
        res_batch_list = []
        # 假设所有样本都有相同的时间步数量
        for i in range(len(batch_list[0])):
            # 为当前时间步 `i` 创建一个批次
            batch = [sample[i] for sample in batch_list]
            new_batch = {}

            # 1. 稀疏体素特征批处理
            input_dict = {
                "coords": [sample.pop('voxel_coordinates') for sample in batch], 
                "feats": [sample.pop('voxel_features') for sample in batch],
            }
            voxel_coordinates, voxel_features = ME.utils.sparse_collate(**input_dict, dtype=torch.int32)
            new_batch['voxel_coordinates'] = voxel_coordinates
            new_batch['voxel_features'] = voxel_features
            
            # 2. 加载和处理离线掩码
            if self.offline_mask_source is not None:
                if self.offline_mask_source == 'gt':
                    padded_segment_masks, padding_mask = pad_sequence_2d([sample['segment_masks'] for sample in batch], return_mask=True)
                    new_batch['offline_attn_mask'] = padded_segment_masks.logical_not()
                    
                    labels = pad_sequence([sample['instance_labels'] for sample in batch], pad=-100)
                    new_batch['target_labels'] = labels
                    new_batch['target_masks'] = padded_segment_masks.float()
                    new_batch['target_masks_pad_masks'] = padding_mask.logical_not()
                else: 
                    raise NotImplementedError(f'{self.offline_mask_source} is not implemented')
                
            # 3. 列表批处理
            list_keys = ['voxel2segment', 'coordinates', 'voxel_to_full_maps', 'segment_to_full_maps', 'raw_coordinates', 'instance_ids', 'instance_labels', 'instance_boxes', 'instance_ids_ori', 'full_masks', 'segment_masks', 'scan_id', 'segment_labels', 'query_selection_ids']
            for k in list_keys:
                new_batch[k] = [sample.pop(k) for sample in batch]

            # 4. 填充批处理
            padding_keys = ['coord_min', 'coord_max', 'obj_center', 'obj_pad_masks', 'seg_center', 'seg_pad_masks', 'seg_point_count', 'query_locs', 'query_pad_masks',
                            'voxel_seg_pad_masks', 'mv_seg_fts', 'mv_seg_pad_masks', 'pc_seg_fts', 'pc_seg_pad_masks', 'prompt', 'prompt_pad_masks']
            padding_keys =  [k for k in padding_keys if k in batch[0].keys()]
            for k in padding_keys:
                tensors = [sample.pop(k) for sample in batch]
                padded_tensor = pad_sequence(tensors)
                new_batch[k] = padded_tensor
        
            # 5. 默认批处理
            new_batch.update(default_collate(batch))
            
            # 将处理好的当前时间步的批次添加到结果列表中
            res_batch_list.append(new_batch)
            
        # 返回一个字典，其中包含所有时间步的批次列表
        return {'batch_list': res_batch_list, 'frame_id': 0, 'total_frame_num': len(res_batch_list)}
