"""
================================================================================
                        Mask3D 3D场景理解模型文件
================================================================================

文件概述:
    本文件实现了基于Mask3D框架的3D场景理解模型，支持实例分割、语义分割等任务。

核心架构:
    - Mask3D: 基础的3D实例分割模型，处理点云体素化特征
    - Mask3DSegLevel: 增强版本，支持多模态输入（文本提示、多视图、点云、体素）
    - CoordinateEncoder: 坐标编码器，将3D坐标转换为位置编码特征

主要功能:
    1. 多模态特征编码: 支持文本提示、多视图图像、点云、体素等多种输入模态
    2. 统一编码: 通过统一编码器融合多模态特征，进行特征交互
    3. 实例分割: 生成实例级别的掩码和类别预测
    4. 零样本泛化: 支持开放词汇设置，可以识别新类别
    5. 位置编码: 使用傅里叶位置编码表示3D空间信息

网络流程:
    输入 → 体素化 → 特征编码 → 统一编码器 → 任务头 → 输出预测

输入数据格式:
    - voxel_features: 体素特征 (N, C)，包含RGB和XYZ坐标
    - voxel_coordinates: 体素坐标 (N, 3)
    - segment/obj相关数据: 分段或目标的相关信息
    - 可选: 文本提示、多视图特征、点云特征等

输出格式:
    data_dict['output'] = {
        'pred_logits': 类别预测 (B, Q, C)
        'pred_masks': 实例掩码 (B, Q, N_segments)
        'aux_outputs': 辅助损失输出
        'sampled_coords': 采样的查询坐标
        'queries': 最终的查询特征
    }

================================================================================
"""

# 导入标准库
from contextlib import nullcontext  # 用于条件语境管理
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络基础模块
from torch.cuda.amp import autocast  # 自动混合精度训练
from torch.nn import functional as F  # 函数式神经网络操作

# 导入第三方库
from torch_scatter import scatter_max, scatter_mean, scatter_min  # 分散操作
import MinkowskiEngine as ME  # 稀疏张量处理库
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling  # 最小斯基引擎池化

# 导入项目内部模块
from model.build import MODEL_REGISTRY, BaseModel  # 模型注册表和基类
from optim.utils import no_decay_param_group  # 优化器参数分组工具

import modules.third_party.mask3d as mask3d_models  # Mask3D第三方模型
from modules.third_party.mask3d.common import conv  # 卷积层
from modules.third_party.mask3d.helpers_3detr import GenericMLP  # 通用MLP层
from modules.third_party.mask3d.position_embedding import PositionEmbeddingCoordsSine  # 傅里叶位置编码
from modules.build import build_module, build_module_by_name  # 模块建构函数
from functools import partial  # 偏函数工具
from copy import deepcopy  # 深拷贝
from transformers import BertTokenizer, T5Tokenizer, AutoTokenizer  # Tokenizer模型
from data.datasets.constant import PromptType  # 提示类型常数
from data.datasets.constant import CLASS_LABELS_200, CLASS_LABELS_REPLICA  # 类别标签
from modules.utils import calc_pairwise_locs  # 成对位置计算工具

# 辅助函数：最远点采样（需要从外部导入或定义）
try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample
except:
    # 如果没有安装，可能在其他地方定义
    pass

@MODEL_REGISTRY.register()
class Mask3D(BaseModel):
    """
    Mask3D 3D实例分割模型（基础版本）
    
    这是基于Mask3D框架的3D场景理解模型，用于实例分割任务。
    支持体素化点云编码，通过Transformer进行多尺度特征融合。
    
    模型流程:
        1. 体素编码: 将点云转换为稀疏张量并提取多层特征
        2. 查询生成: 使用最远点采样或GT坐标生成查询
        3. 统一编码: 使用Transformer融合多尺度特征
        4. 掩码生成: 预测实例掩码和类别
    
    属性:
        cfg: 配置对象
        use_gt_mask: 是否使用GT真实值掩码（用于训练）
        voxel_encoder: 体素编码器
        unified_encoder: 统一编码器
        mask_head: 掩码预测头
        query_projection: 查询投影层
        pos_enc: 位置编码器
    """
    
    def __init__(self, cfg):
        """
        初始化Mask3D模型。
        
        参数:
            cfg: 配置对象，包含模型超参数
        """
        super().__init__(cfg)
        self.cfg = cfg
        
        # 是否在训练时使用GT掩码（主要用于调试或特定的训练策略）
        self.use_gt_mask = cfg.model.get("use_gt_mask", False)
        
        # ========== 构建体素编码器 ==========
        # 遍历所有输入模态并构建对应的编码器
        for input in cfg.model.inputs:
            encoder = input + '_encoder'
            self.encoder = encoder
            setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        
        # ========== 构建统一编码器 ==========
        # 用于融合多尺度特征和进行交叉模态交互的Transformer
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        
        # ========== 构建任务头 ==========
        # 当前只支持在分段级别的分割
        self.seg_on_segments = cfg.model.seg_on_segments
        assert self.seg_on_segments == True  # 只支持分段级别的分割
        # 为每个任务头（如mask_head）构建模块
        for head in self.cfg.model.heads.head_list:
            setattr(self, head, build_module("heads", getattr(self.cfg.model.heads, head)))
            
        # ========== 构建查询相关模块 ==========
        # 查询数量
        self.num_queries = cfg.model.num_queries
        # 隐层维度（即特征维度）
        hidden_size = cfg.model.voxel_encoder.args.query_dim
        # 查询投影网络：将编码后的特征投影到合适的维度
        self.query_projection = GenericMLP(
            input_dim=hidden_size,
            hidden_dims=[hidden_size],
            output_dim=hidden_size,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        # 位置编码器：使用傅里叶编码方式将3D坐标转换为特征
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=hidden_size,
                                                       gauss_scale=1.0,
                                                       normalize=True)
         
    def forward(self, data_dict):
        """
        前向传播。
        
        参数:
            data_dict: 包含以下关键信息：
                - voxel_features: 体素特征 (N, C)
                - voxel_coordinates: 体素坐标 (N, 3)
                - segment_to_voxel_maps: 分段到体素的映射
                - obj_center: 目标中心坐标（GT坐标）
                - segment_masks: 分段掩码（如果use_gt_mask=True）
        
        返回:
            data_dict: 更新后的数据字典，包含预测结果
        """
        # ========== 准备体素数据 ==========
        voxel_features = data_dict['voxel_features']  # RGB + XYZ
        voxel_coordinates = data_dict['voxel_coordinates']
        # 创建稀疏张量，特征去掉最后3维的坐标（已在coordinates中）
        x = ME.SparseTensor(coordinates=voxel_coordinates, features=voxel_features[:, :-3], device=voxel_features.device)

        # ========== 准备Swin3D特定的输入（如果使用Swin3D编码器） ==========
        if self.cfg.model.get(self.encoder, None).get('signal', False):
            # 如果需要使用信号特征
            if voxel_features.shape[1] > 3:
                if self.cfg.model.get(self.encoder, None).get('use_offset', False):
                    # 计算体素位置偏移量
                    voxel_features[:, -3:] = voxel_coordinates[:, -3:] - voxel_coordinates[:, -3:].int()
            swin_sp = ME.SparseTensor(coordinates=voxel_coordinates.int(), features=voxel_features, device=voxel_features.device)
        else:
            # 否则使用全1特征
            swin_sp = ME.SparseTensor(coordinates=voxel_coordinates.int(), features=torch.ones_like(voxel_features).float(), device=voxel_features.device)
        
        # 准备颜色特征（归一化）
        colors = voxel_features[:, 0:3] / 1.001
        # 创建包含坐标和颜色的稀疏张量
        swin_coords_sp = ME.SparseTensor(
            features=torch.cat([voxel_coordinates, colors], dim=1), 
            coordinate_map_key=swin_sp.coordinate_map_key, 
            coordinate_manager=swin_sp.coordinate_manager
        )
        
        # 获取点到分段的映射
        point2segment = data_dict['segment_to_voxel_maps']
        
        # ========== 体素编码（骨干网络） ==========
        with self.optional_freeze():
            if 'Swin3D' in self.cfg.model.get(self.encoder, None).name:
                # 如果使用Swin3D编码器
                mask_features, mask_segments, coordinates, multi_scale_features, multi_scale_coordinates = self.voxel_encoder(
                    swin_sp, swin_coords_sp, voxel_features, point2segment, voxel_coordinates[:, 0]
                )
            else:
                # 使用其他编码器（如MinkowskiNet）
                mask_features, mask_segments, coordinates, multi_scale_features, multi_scale_coordinates = self.voxel_encoder(
                    x, voxel_features, point2segment
                )
        
        # ========== 构建掩码头的偏函数 ==========
        if self.use_gt_mask:
            # 如果使用GT掩码，传入GT掩码用于指导
            mask_head_partial = partial(self.mask_head, mask_features=mask_features, mask_segments=mask_segments, 
                                       ret_attn_mask=True, point2segment=point2segment, 
                                       gt_attn_mask=data_dict['segment_masks'])
        else:
            # 否则不使用GT掩码
            mask_head_partial = partial(self.mask_head, mask_features=mask_features, mask_segments=mask_segments, 
                                       ret_attn_mask=True, point2segment=point2segment)
        
        # ========== 构建位置编码 ==========
        pos_encodings_pcd = self.get_multi_level_pos_encs(multi_scale_coordinates)
        
        # ========== 构建查询（Query） ==========
        sampled_coords = None
        if self.use_gt_mask:
            # 使用GT坐标作为查询坐标
            gt_coordinates = deepcopy(data_dict['obj_center'])
            # 初始化查询掩码（True表示填充）
            query_masks = torch.ones((len(gt_coordinates), self.num_queries), dtype=torch.bool,
                                    device=gt_coordinates[0].get_device())
            # 标记有效查询（False表示有效）
            for bid in range(len(gt_coordinates)):
                query_masks[bid, : gt_coordinates[bid].shape[0]] = False
            
            # 填充查询到指定数量
            for bid in range(len(gt_coordinates)):
                assert gt_coordinates[bid].shape[0] < self.num_queries
                if gt_coordinates[bid].shape[0] < self.num_queries:
                    # 用零填充不足的查询
                    gt_coordinates[bid] = torch.cat([
                        gt_coordinates[bid], 
                        torch.zeros(self.num_queries - gt_coordinates[bid].shape[0], 3)
                            .to(gt_coordinates[bid].get_device())
                    ], dim=0)
                else:
                    # 随机采样查询
                    perm = torch.randperm(gt_coordinates[bid].shape[0])[:self.num_queries]
                    gt_coordinates[bid] = gt_coordinates[bid][perm]
            sampled_coords = torch.stack(gt_coordinates)
        else:
            # 使用最远点采样获取查询坐标
            gt_coordinates = deepcopy(data_dict['obj_center'])
            # 初始化查询掩码（False表示有效）
            query_masks = torch.zeros((len(gt_coordinates), self.num_queries), dtype=torch.bool,
                                     device=gt_coordinates[0].get_device())
            # 最远点采样
            fps_idx = [furthest_point_sample(x.decomposed_coordinates[i][None, ...].float(),
                                            self.num_queries).squeeze(0).long()
                    for i in range(len(x.decomposed_coordinates))]
            sampled_coords = torch.stack([coordinates.decomposed_features[i][fps_idx[i].long(), :]
                                        for i in range(len(fps_idx))])
        
        # 获取坐标范围用于位置编码归一化
        mins = torch.stack([coordinates.decomposed_features[i].min(dim=0)[0] 
                           for i in range(len(coordinates.decomposed_features))])
        maxs = torch.stack([coordinates.decomposed_features[i].max(dim=0)[0] 
                           for i in range(len(coordinates.decomposed_features))])
        
        # 对采样坐标进行位置编码
        query_pos = self.pos_enc(sampled_coords.float(), input_range=[mins, maxs])  # (B, D, Q)
        query_pos = self.query_projection(query_pos).permute(0, 2, 1)  # (B, Q, D)
        
        # 初始化查询特征为零
        queries = torch.zeros_like(query_pos)
        
        # ========== 统一编码（多尺度Transformer） ==========
        queries, predictions_class, predictions_mask = self.unified_encoder(
            queries, query_pos, multi_scale_features, pos_encodings_pcd, mask_head_partial, 
            not self.training, query_masks
        )
        
        # ========== 最后的掩码头 ==========
        output_class, outputs_mask = self.mask_head(
            query_feat=queries, mask_features=mask_features, mask_segments=mask_segments, 
            num_pooling_steps=0, ret_attn_mask=False, point2segment=point2segment
        )
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)
    
        # ========== 整理输出 ==========
        data_dict['output'] = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask),
            'sampled_coords': sampled_coords.detach().cpu().numpy() if sampled_coords is not None else None,
            'queries': queries
        }
        return data_dict

    def get_opt_params(self):
        """获取优化器参数组。"""
        return [{'params': self.parameters(), 'lr': self.cfg.solver.lr}]

    def get_multi_level_pos_encs(self, coords):
        """
        获取多尺度位置编码。
        
        参数:
            coords: 多尺度坐标列表，每个元素对应一个尺度的坐标
        
        返回:
            pos_encodings_pcd: 多尺度位置编码列表
        """
        pos_encodings_pcd = []

        # 遍历每个尺度
        for i in range(len(coords)):
            pos_encodings_pcd.append([])
            # 遍历每个批次
            for coords_batch in coords[i].decomposed_features:
                # 获取该批次的坐标范围
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                # 用autocast禁用混合精度以确保数值精度
                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(),
                                       input_range=[scene_min, scene_max])

                # 调整维度顺序：(1, N, D) -> (N, D)
                pos_encodings_pcd[-1].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        """
        设置辅助损失。
        
        这是一个workaround，用于兼容TorchScript。
        返回中间层的预测结果用于辅助监督。
        
        参数:
            outputs_class: 所有层的类别预测
            outputs_seg_masks: 所有层的掩码预测
        
        返回:
            辅助损失字典列表
        """
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
        
@MODEL_REGISTRY.register()
class Mask3DSegLevel(BaseModel):
    """
    Mask3D 分段级别的3D场景理解模型（增强版本）
    
    这是Mask3D的扩展版本，支持多模态输入（文本提示、多视图、点云、体素等），
    可以处理分段级别的分类和掩码预测任务。
    
    主要特点:
        1. 多模态融合: 支持文本、视觉等多种输入模态
        2. 分段级处理: 直接在分段（segment）级别进行处理，而非点云级别
        3. 开放词汇: 支持文本提示，可以进行零样本或少样本学习
        4. 灵活的查询: 支持GT查询和FPS采样
        5. 条件掩码预测: 支持使用GT掩码进行条件化训练
    
    工作流程:
        输入(多模态) → 特征编码 → 统一编码器 → 掩码生成 → 输出预测
    """
    
    def __init__(self, cfg):
        """
        初始化Mask3DSegLevel模型。
        
        参数:
            cfg: 配置对象
        """
        super().__init__(cfg)
        self.cfg = cfg
        
        # ========== 基本配置 ==========
        # 记忆（输入模态）列表
        self.memories = cfg.model.memories
        # 任务头列表
        self.heads = cfg.model.heads
        # 查询数量
        self.num_queries = cfg.model.num_queries
        # 隐层维度
        self.hidden_size = cfg.model.hidden_size
        # 训练时是否使用GT掩码
        self.use_gt_mask = cfg.model.get("use_gt_mask", False)
        # 评估时是否使用GT掩码
        self.use_gt_mask_eval = cfg.eval.get("use_gt_mask", False)
        # 是否使用离线预计算的体素特征
        self.use_offline_voxel_fts = cfg.model.get("use_offline_voxel_fts", False)

        # 输入类型列表
        self.inputs = self.memories[:]
        
        # ========== 提示类型和编码器 ==========
        # 当前只支持文本提示
        self.prompt_types = ['txt']
        # 为每个输入模态构建编码器
        for input in self.inputs:
            if input == 'prompt':  # 文本提示编码器
                for prompt_type in self.prompt_types:
                    encoder = prompt_type + '_encoder'
                    setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
            else:  # 其他模态编码器
                encoder = input + '_encoder'
                setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        
        # ========== 统一编码器 ==========
        # 融合多模态特征的Transformer
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        
        # ========== 任务头 ==========
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))
            
        # ========== 坐标编码器和空间位置关系 ==========
        hidden_size = self.hidden_size
        # 坐标编码器：将3D坐标转换为特征向量
        self.coord_encoder = CoordinateEncoder(hidden_size)
        # 成对位置关系类型（如'center', 'relative'等）
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type if hasattr(self.cfg.model, 'obj_loc') else 'center' 
        # 空间维度（通常为3D或5D）
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim if hasattr(self.cfg.model, 'obj_loc') else 5

        # ========== 零样本设置 ==========
        # 是否在测试时使用ScanNet200标签进行提示
        self.test_prompt_scannet200 = cfg.eval.get("test_prompt_scannet200", False)
        # 是否在测试时使用Replica标签进行提示
        self.test_prompt_replica = cfg.eval.get("test_prompt_replica", False)
        # 需要过滤掉的类别（如label 0: unlabeled）
        self.filter_out_classes = cfg.eval.get("filter_out_classes", [0, 2])
        # 初始化tokenizer用于文本编码
        if self.test_prompt_scannet200 or self.test_prompt_replica:
            self.tokenizer = AutoTokenizer.from_pretrained("/home/ma-user/work/zhangWei/mtu3d/data/trans/clip-vit-large-patch14")

    def prompt_encoder(self, data_dict):
        """
        对文本提示进行编码。
        
        参数:
            data_dict: 包含以下键：
                - prompt: 文本提示 (B, text_len)
                - prompt_masks: 提示掩码
                - prompt_type: 提示类型（Txt, Image等）
        
        返回:
            prompt_feat: 编码后的提示特征 (B, text_len, hidden_size)
            prompt_mask: 反向的掩码（True表示有效）
        """
        prompt = data_dict['prompt']
        prompt_mask = data_dict['prompt_masks']
        prompt_type = data_dict['prompt_type']
        # 初始化提示特征
        prompt_feat = torch.zeros(prompt.shape + (self.hidden_size,), device=prompt.device)
        
        # 遍历每种提示类型
        for type in self.prompt_types:
            # 获取该类型的编码器
            encoder = getattr(self, type + '_encoder')
            # 获取该类型提示的索引
            idx = prompt_type == getattr(PromptType, type.upper())
            input = prompt[idx]
            mask = prompt_mask[idx]
            
            # 编码提示
            if type == 'txt':
                # 文本编码器输入：token ids和mask
                feat = encoder(input.long(), mask)
            else:
                raise NotImplementedError
            
            # 存储编码结果
            prompt_feat[idx] = feat
        
        # 返回特征和反向掩码（True表示有效）
        return prompt_feat, prompt_mask.logical_not()

    def forward(self, data_dict):
        """
        前向传播。
        
        参数:
            data_dict: 包含所有输入数据
        
        返回:
            data_dict: 更新后的数据字典，包含预测结果
        """
        # ========== 准备数据 ==========
        voxel_features = data_dict['voxel_features']
        voxel_coordinates = data_dict['voxel_coordinates']
        # 创建稀疏张量
        x = ME.SparseTensor(coordinates=voxel_coordinates, features=voxel_features[:, :-3], device=voxel_features.device)
        voxel2segment = data_dict['voxel2segment']
        coordinates = data_dict['coordinates']
        coord_min = data_dict['coord_min']
        coord_max = data_dict['coord_max']
        seg_center = data_dict['seg_center']
        
        # ========== 决定是否使用GT掩码 ==========
        # 训练时根据use_gt_mask，评估时根据use_gt_mask_eval
        use_gt_mask = (self.training and self.use_gt_mask) or (not self.training and self.use_gt_mask_eval)
        data_dict['use_gt_mask'] = use_gt_mask
        
        # ========== 准备分段级别的特征和掩码 ==========
        # 分段有效性掩码
        seg_pad_masks = data_dict['seg_pad_mask'].logical_not()
        # 分段位置编码
        seg_pos = self.coord_encoder(seg_center, input_range=[coord_min, coord_max])
        # 多视图分段掩码（结合pad_mask和原始mask）
        mv_seg_pad_masks = torch.logical_or(seg_pad_masks, data_dict['mv_seg_pad_mask'].logical_not())
        # 点云分段掩码
        pc_seg_pad_masks = torch.logical_or(seg_pad_masks, data_dict['pc_seg_pad_mask'].logical_not())

        # ========== 编码多个输入模态 ==========
        input_dict = {}
        for input in self.inputs:
            feat, mask, pos = None, None, None
            
            if input == 'prompt':  # 文本提示编码
                feat, mask = self.prompt_encoder(data_dict)
                
            elif input == 'mv':  # 多视图特征编码
                feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                mask = mv_seg_pad_masks
                pos = seg_pos
                
            elif input == 'pc':  # 点云特征编码
                feat = self.pc_encoder(obj_feats = data_dict['pc_seg_fts'])
                mask = pc_seg_pad_masks
                pos = seg_pos
                
            elif input == 'voxel':  # 体素特征编码
                if self.use_offline_voxel_fts:
                    # 使用预计算的离线体素特征
                    feat = data_dict['voxel_seg_fts']
                else:
                    # 实时编码体素特征
                    feat = self.voxel_encoder(x, voxel2segment, max_seg=seg_center.shape[1])
                # 保存体素特征供后续使用
                voxel_seg_feature = feat.copy()
                mask = seg_pad_masks
                pos = seg_pos
            else:
                raise NotImplementedError(f"Unknow input type: {input}")
            
            # 将(特征, 掩码, 位置)存储到input_dict
            input_dict[input] = [feat, mask, pos]
            
        # ========== 标准模式处理（非开放词汇测试） ==========
        if not self.test_prompt_scannet200 or self.training:
            # 构建查询
            if use_gt_mask:
                # 使用GT目标作为查询
                query_masks = data_dict['obj_pad_mask'].logical_not()
                sampled_coords = data_dict['obj_center']
            else:
                # 使用FPS采样获取查询
                query_masks = None
                voxel_coordinates = x.decomposed_coordinates
                fps_idx = [furthest_point_sample(voxel_coordinates[i][None, ...].float(), self.num_queries).squeeze(0).long()
                        for i in range(len(voxel_coordinates))]
                sampled_coords = torch.stack([coordinates[i][fps_idx[i]] for i in range(len(fps_idx))])
            
            # 对采样坐标进行位置编码
            query_pos = self.coord_encoder(sampled_coords, input_range=[coord_min, coord_max])
            # 初始化查询特征
            query = torch.zeros_like(query_pos)
            input_dict['query'] = [query, query_masks, query_pos]
            
            # ========== 构建空间注意力的成对位置关系 ==========
            if self.unified_encoder.spatial_selfattn:
                pairwise_locs = calc_pairwise_locs(sampled_coords, None, pairwise_rel_type=self.pairwise_rel_type, 
                                                   spatial_dist_norm=True, spatial_dim=self.spatial_dim)
            else:
                pairwise_locs = None
                
            # ========== 用于分段匹配的特征 ==========
            seg_fts_for_match = []
            for input in self.inputs:
                if input in ['voxel', 'mv', 'pc']:
                    feats = input_dict[input][:]
                    if isinstance(feats[0], list):
                        # 体素特征是多层的，使用最后一层
                        assert input == 'voxel'
                        feats[0] = feats[0][-1]
                    seg_fts_for_match.append(feats)
            
            # ========== 构建掩码头的偏函数 ==========
            gt_attn_mask = data_dict['gt_attn_mask'] if use_gt_mask else None
            mask_head_partial = partial(self.mask_head, seg_fts_for_match=seg_fts_for_match, 
                                        seg_pad_masks=seg_pad_masks, gt_attn_mask=gt_attn_mask, 
                                        query_pos=query_pos)

            # ========== 统一编码 ==========
            query, predictions_class, predictions_mask = self.unified_encoder(input_dict, pairwise_locs, 
                                                                              mask_head_partial)

            # ========== 接地（Grounding）头处理 ==========
            if hasattr(self, 'ground_head'):
                ground_logits = self.ground_head(query, query_masks)
                data_dict['ground_logits'] = ground_logits
                
        # ========== 开放词汇测试模式 ==========
        else:
            # 获取类别标签总数
            if self.test_prompt_scannet200:
                class_label = [f"a {class_name} in a scene" for class_name in CLASS_LABELS_200]
            elif self.test_prompt_replica:
                class_label = [f"a {class_name} in a scene" for class_name in CLASS_LABELS_REPLICA]
            total_number = len(class_label) + 1  # +1用于空类别
            
            # ========== 构建所有类别的提示 ==========
            prompt = []
            for class_name in class_label:
                prompt.append(class_name)
            prompt.append("")  # 空提示
            
            # Tokenize并编码提示
            encoded_input = self.tokenizer(prompt, padding='max_length', return_tensors="pt", 
                                          truncation=True, max_length=50)
            prompt_feats = self.txt_encoder(encoded_input.input_ids.to(voxel_features.device), 
                                           encoded_input.attention_mask.bool().to(voxel_features.device))
            prompt_mask = encoded_input.attention_mask.bool().logical_not().to(voxel_features.device) 
            input_dict['prompt'] = [prompt_feats, prompt_mask, None]
            
            # ========== 扩展数据以处理所有类别 ==========
            if use_gt_mask:
                query_masks = self.expand_tensor(data_dict['obj_pad_mask'].logical_not(), total_number)
                sampled_coords = data_dict['obj_center']
            else:
                query_masks = None
                voxel_coordinates = x.decomposed_coordinates
                fps_idx = [furthest_point_sample(voxel_coordinates[i][None, ...].float(), self.num_queries).squeeze(0).long()
                        for i in range(len(voxel_coordinates))]
                sampled_coords = torch.stack([coordinates[i][fps_idx[i]] for i in range(len(fps_idx))])
            
            # 扩展查询位置编码和初始特征
            query_pos = self.expand_tensor(self.coord_encoder(sampled_coords, input_range=[coord_min, coord_max]), total_number)
            query = self.expand_tensor(torch.zeros_like(query_pos), total_number)
            input_dict['query'] = [query, query_masks, query_pos]
            
            # ========== 扩展空间位置关系 ==========
            if self.unified_encoder.spatial_selfattn:
                pairwise_locs = self.expand_tensor(calc_pairwise_locs(sampled_coords, None, 
                                                                       pairwise_rel_type=self.pairwise_rel_type, 
                                                                       spatial_dist_norm=True, 
                                                                       spatial_dim=self.spatial_dim), total_number)
            else:
                pairwise_locs = None
                
            # ========== 扩展用于分段匹配的特征 ==========
            seg_fts_for_match = []
            for input in self.inputs:
                if input in ['voxel', 'mv', 'pc']:
                    feats = input_dict[input][:]
                    if isinstance(feats[0], list):
                        assert input == 'voxel'
                        feats[0] = feats[0][-1]
                    # 扩展到所有类别数
                    feats[0] = self.expand_tensor(feats[0], total_number)
                    seg_fts_for_match.append(feats)
            
            # ========== 构建掩码头的偏函数 ==========
            gt_attn_mask = data_dict['gt_attn_mask'].repeat_interleave(total_number, 0) if use_gt_mask else None
            mask_head_partial = partial(self.mask_head, seg_fts_for_match=seg_fts_for_match, 
                                        seg_pad_masks=seg_pad_masks.repeat_interleave(total_number, 0), 
                                        gt_attn_mask=gt_attn_mask, query_pos=query_pos)

            # ========== 扩展所有输入 ==========
            for input in self.inputs:
               for idx in range(3):
                   if input == 'voxel' and idx == 0:
                       # 体素特征是列表，需要逐层扩展
                       for l in range(len(input_dict[input][idx])):
                           input_dict[input][idx][l] = self.expand_tensor(input_dict[input][idx][l], total_number)
                   else:
                       input_dict[input][idx] = self.expand_tensor(input_dict[input][idx], total_number)

            # ========== 统一编码 ==========
            query, predictions_class, predictions_mask = self.unified_encoder(input_dict, pairwise_locs, 
                                                                              mask_head_partial)

            # ========== 接地头处理 ==========
            if hasattr(self, 'ground_head'):
                ground_logits = self.ground_head(query, query_masks)
                # 只保留实际类别的接地得分，去掉扩展部分
                data_dict['ground_logits'] = ground_logits[total_number -1][None, ...]
                
            # ========== 重新组织接地结果 ==========
            prompt_logits = ground_logits.permute(1, 0)[None, ..., :total_number - 1]
            # 过滤掉指定的类别（如unlabeled）
            for filter_out_id in self.filter_out_classes:
                prompt_logits[..., filter_out_id] = float("-inf")
            prompt_masks = [predictions_mask[-1].permute(1, 2, 0)[..., :total_number-1]]
                
                
        # ========== 整理输出 ==========
        data_dict['output'] = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'prompt_logits': prompt_logits if (self.test_prompt_scannet200 or self.test_prompt_replica) and not self.training else None,
            'prompt_masks': prompt_masks if (self.test_prompt_scannet200 or self.test_prompt_replica) and not self.training else None,
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask),
            'voxel_seg_feature': voxel_seg_feature, 
            'voxel_seg_mask': seg_pad_masks
        }
        return data_dict

    def get_opt_params(self):
        """
        获取优化器参数组。
        
        返回:
            参数组列表，包含模型所有参数和学习率
        """
        return [{'params': self.parameters(), 'lr': self.cfg.solver.lr}]
    
    def expand_tensor(self, x, num):
        """
        扩展张量以处理多个类别。
        
        用于在开放词汇设置中，将单个体素特征扩展为多个类别的副本。
        
        参数:
            x: 输入张量或None
            num: 扩展的倍数（类别数）
        
        返回:
            扩展后的张量或None
        
        示例:
            x.shape = (B, D, N) -> expand_tensor(x, 3).shape = (3*B, D, N)
        """
        return x.expand([num] + [-1] * (len(x.shape) - 1)) if x is not None else None

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        """
        设置辅助损失（用于中间层的监督）。
        
        这是一个workaround，用于兼容TorchScript。
        返回中间层的预测结果用于辅助损失计算。
        
        参数:
            outputs_class: 所有层的类别预测列表
            outputs_seg_masks: 所有层的掩码预测列表
        
        返回:
            辅助损失字典列表（去掉最后一层）
        """
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
        

class CoordinateEncoder(nn.Module):
    """
    坐标编码器（位置编码）。
    
    将3D坐标转换为特征向量，使用傅里叶位置编码。
    主要用于为3D空间中的点或对象提供位置信息。
    
    功能:
        1. 傅里叶位置编码: 将坐标转换为周期性特征
        2. 特征投影: 使用线性层进行特征投影和归一化
        3. 范围归一化: 支持基于坐标范围的归一化
    
    工作流程:
        坐标 → 傅里叶编码 → 维度调整 → 可选投影 → 特征输出
    """
    
    def __init__(self, hidden_size, use_projection=True):
        """
        初始化坐标编码器。
        
        参数:
            hidden_size: 输出特征维度
            use_projection: 是否使用投影层
        """
        super().__init__()
        # 傅里叶位置编码器
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",  # 使用傅里叶编码
                                                    d_pos=hidden_size,  # 输出维度
                                                    gauss_scale=1.0,  # 高斯尺度
                                                    normalize=True)  # 是否归一化
        
        if use_projection:
            # 投影网络：线性投影 + 层归一化
            self.feat_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
    
    def forward(self, coords, input_range):
        """
        前向传播。
        
        参数:
            coords: 输入坐标，形状为 (B, N, 3)
            input_range: 坐标范围，[coord_min, coord_max] 用于归一化
        
        返回:
            编码后的特征，形状为 (B, N, hidden_size)
        """
        # 禁用混合精度以确保数值精度
        with autocast(enabled=False):
            # 傅里叶编码，返回 (B, hidden_size, N)
            pos = self.pos_enc(coords, input_range=input_range).permute(0, 2, 1)  # (B, N, hidden_size)
        
        # 可选的特征投影
        if hasattr(self, 'feat_proj'):
            pos = self.feat_proj(pos)
        
        return pos