"""
================================================================================
                       EmbodiedPQ3D 实例分割模型文件
================================================================================

文件概述：
    本文件实现 `EmbodiedPQ3DInstSegModel`，用于在具身（embodied）
    三维探索场景中的实例分割任务。模型为多模态架构，能够融合
    体素（voxel）、多视图（mv）等特征，并提供掩码头与开放词汇头等
    多种任务接口。

主要组件与流程：
    - `scatter_norm`：对同一分段内点云进行规范化（单位直径球体）。
    - 特征编码器（按 `cfg.model.memories` 构建）：为每种输入模态构建编码器。
    - `CoordinateEncoder`：坐标/位置编码器，用于将3D坐标转换为特征。
    - 统一编码器（`unified_encoder`）：融合所有模态的特征并执行交互注意力。
    - 任务头（例如 `mask_head`, `openvocab_head`）：分别负责掩码预测和开放词汇任务。

数据流（简要）：
    输入 data_dict -> 构建 query（位置编码/可选特征初始化） -> 各模态特征编码 ->
    生成空间成对位置（可选） -> 统一编码器融合 -> 任务头输出 -> 更新 data_dict

输入输出说明（常见键）：
    输入: `query_pad_masks`, `query_locs`, `seg_center`, `mv_seg_fts`, `voxel_features`, `voxel_coordinates`, ...
    输出: data_dict['predictions_score/class/mask/box'] 或 data_dict['openvocab_query_feat'] 等

================================================================================
"""

# 导入必要的库
from copy import copy  # 用于对象复制
from functools import partial  # 用于创建偏函数
import torch  # PyTorch库
import torch.nn as nn  # PyTorch神经网络模块
import MinkowskiEngine as ME  # 稀疏张量处理库

# 导入数据集相关模块
from data.datasets.constant import PromptType

# 导入模型建构和工具函数
from modules.build import build_module_by_name  # 模块构建函数
from modules.utils import calc_pairwise_locs  # 计算成对位置关系
from model.build import MODEL_REGISTRY, BaseModel  # 模型注册表和基类
from optim.utils import no_decay_param_group  # 优化器参数分组
from model.mask3d import CoordinateEncoder  # 坐标编码器
from torch_scatter import scatter_mean, scatter  # 分散操作
from torch.nn.utils.rnn import pad_sequence  # 序列填充函数

def scatter_norm(points, idx):
    """
    将相同段内的点位置规范化到直径为1的单位球体内。
    
    参数:
        points: 输入点的坐标，形状为 (N, 3)
        idx: 点所属的段索引，形状为 (N,)
    
    返回:
        points: 规范化后的点坐标
        diameter_segment: 每个段的直径，形状为 (num_segments, 1)
    """
    # 计算每个段内点的最小坐标
    min_segment = scatter(points, idx, dim=0, reduce='min')
    # 计算每个段内点的最大坐标
    max_segment = scatter(points, idx, dim=0, reduce='max')
    # 计算每个段的直径（最大坐标与最小坐标的差的最大值）
    diameter_segment = (max_segment - min_segment).max(dim=1).values
    # 计算每个段的中心（以平均值作为中心）
    center_segment = scatter(points, idx, dim=0, reduce='mean')
    # 为每个点分配其所属段的中心
    center = center_segment[idx]
    # 为每个点分配其所属段的直径
    diameter = diameter_segment[idx]
    # 将点规范化：先减去中心再除以直径（加上小的epsilon防止除以零）
    points = (points - center) / (diameter.view(-1, 1) + 1e-2)
    return points, diameter_segment.view(-1, 1)

@MODEL_REGISTRY.register()
class EmbodiedPQ3DInstSegModel(BaseModel):
    """
    EmbodiedPQ3D 实例分割模型
    
    这是一个多模态 3D 场景理解模型，用于处理具实空间探索场景中的实例分割任务。
    支持多种输入模态（多视图、体素等）和多个任务头（掩码、开放词汇等）。
    """
    
    def __init__(self, cfg):
        """
        初始化 EmbodiedPQ3D 模型。
        
        参数:
            cfg: 配置对象，包含模型的所有超参数配置
        """
        super().__init__(cfg)
        
        # ========== 配置参数设置 ==========
        # 记录配置
        self.cfg = cfg
        # 获取内存（输入模态）列表，如['voxel', 'mv']
        # memories 定义模型支持的 "记忆" 模态列表，用于统一编码器的多模态融合。
        # 语义：每个 memory 对应一个模态的特征集合（例如 'voxel' -> 体素特征列表, 'mv' -> 多视图特征）
        # 注意：编码器需要输出与 hidden_size 对齐的特征，统一编码器会按 memory 名称查找对应输入
        self.memories = cfg.model.memories
        # 获取任务头列表，如['mask', 'openvocab']
        self.heads = cfg.model.heads
        # 复制内存列表作为输入类型
        self.inputs = self.memories[:]
        # 空间关系类型（如'relative', 'absolute'等）
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type
        # 空间维度（通常为3维）
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim
        # 多头自注意力中的头数
        self.num_heads = self.cfg.model.unified_encoder.args.num_attention_heads
        # 是否跳过查询编码器的掩码预测
        self.skip_query_encoder_mask_pred = cfg.model.get('skip_query_encoder_mask_pred', False)
        # 是否通过特征初始化查询
        self.init_query_by_feat = cfg.model.get('init_query_by_feat', False)
        # 是否向分段添加几何特征
        self.add_geometry_to_segment = cfg.model.get('add_geometry_to_segment', False)
        
        # ========== 特征编码器构建 ==========
        # 为每个输入模态（如mvoxel, voxel等）构建对应的编码器
        # 约定：配置中应当存在以 '<memory>_encoder' 命名的模块定义，例如 'voxel_encoder', 'mv_encoder'
        # 每个 encoder 的输出语义如下：
        #  - 对于体素（voxel）：通常返回多尺度特征列表（list of tensors），每个 tensor 形状为 (B, S_i, hidden_size)
        #  - 对于多视图（mv）：通常返回一个张量形状为 (B, S_mv, hidden_size)
        for input in self.inputs:
            encoder = input + '_encoder'
            setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        
        # ========== 坐标编码器构建 ==========
        # 位置编码的维度
        dim_loc = self.cfg.model.obj_loc.dim_loc
        # 模型的隐层维度
        hidden_size = self.cfg.model.hidden_size
        self.dim_loc = dim_loc
        self.hidden_size = hidden_size
        # 创建坐标编码器（用于将3D坐标编码成特征向量）
        self.coord_encoder = CoordinateEncoder(hidden_size)
        
        # ========== 统一编码器构建 ==========
        # 统一编码器用于融合多个模态的特征和处理跨模态交互
        # unified_encoder 的实现应当按照配置中的 memories 列表为每个 memory 提供交叉注意力路径。
        # 它期望的输入格式为一个字典：{ 'query': (feat, mask, pos), 'voxel': [feat_list, mask, pos], 'mv': [feat, mask, pos], ... }
        # 其中对 voxel 模态, feat_list 为多尺度列表（encoder 返回值）；对其他模态 feat 为单一张量。
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        
        # ========== 任务头构建 ==========
        # 为每个任务头（如mask_head, openvocab_head）构建模块
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))
        
        # ========== 几何特征处理模块（可选） ==========
        if self.add_geometry_to_segment:
            # 投影网络1：将3D点坐标投影到隐层维度
            self.pts_proj1 = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            # 特征投影网络：融合特征和几何信息
            self.feat_proj = nn.Sequential(
                nn.Linear(hidden_size + 3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        
        # ========== 查询初始化权重（可选） ==========
        if self.init_query_by_feat:
            # 为不同输入模态定义可学习的权重
            self.input_weights = nn.ParameterDict({
                input: nn.Parameter(torch.ones(1)) for input in self.inputs
            })
         
    def forward(self, data_dict):
        """
        前向传播方法。
        
        参数:
            data_dict: 包含以下关键输入：
                - query_pad_masks: 查询填充掩码
                - query_locs: 查询位置
                - seg_center: 分段中心
                - mv_seg_fts: 多视图分段特征
                - mv_seg_pad_masks: 多视图分段填充掩码
                - voxel_features: 体素特征
                - voxel_coordinates: 体素坐标
                - coord_min/coord_max: 坐标范围
                等等
        
        返回:
            data_dict: 更新后的数据字典，包含模型预测结果
        """
        input_dict = {}
        
        # ========== 构建查询初始化 ==========
        # 从填充掩码反向计算有效掩码（True表示有效）
        mask = data_dict['query_pad_masks'].logical_not()
        # 获取查询位置信息（前dim_loc维）
        query_locs = data_dict['query_locs'][:, :, :self.dim_loc]
        # 获取场景的坐标范围
        coord_min = data_dict['coord_min']
        coord_max = data_dict['coord_max']
        # 使用坐标编码器对查询位置进行编码
        query_pos = self.coord_encoder(query_locs[:, :, :3], input_range=[coord_min, coord_max])
        # 初始化查询特征为零
        feat = torch.zeros_like(query_pos)
        # 位置编码作为位置信息
        pos = query_pos
        # 将查询打包为(特征, 掩码, 位置)的元组
        input_dict['query'] = (feat, mask, pos)
        
        # ========== 编码多个输入模态的特征 ==========
        # 计算分段中心的编码位置
        fts_locs = data_dict['seg_center']
        fts_pos = self.coord_encoder(fts_locs[:, :, :3], input_range=[coord_min, coord_max])
        
        # 遍历处理每个输入模态
        # 遍历每个 memory 输入并构建统一输入字典 input_dict
        # 每个条目 input_dict[input] = [feat, mask, pos]
        # feat 的类型：
        #   - 对于 'voxel'：feat 是 list[tensor]，表示多尺度特征（从低到高或反之，视 encoder 实现）
        #   - 对于 'mv'：feat 是 tensor，表示多视图分段特征 (B, S_seg, hidden_size)
        for input in self.inputs:
            feat, mask, pos = None, None, None
            
            if input == 'mv':  # 多视图模态
                # 使用多视图编码器处理多视图分段特征
                feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                # 计算多视图分段的有效掩码
                mask = data_dict['mv_seg_pad_masks'].logical_not()
                # 位置信息为分段中心的编码位置
                pos = fts_pos
                
            elif input == 'voxel':  # 体素模态
                # 获取体素特征和坐标（MinkowskiEngine 格式的稀疏张量输入）
                voxel_features = data_dict['voxel_features']
                voxel_coordinates = data_dict['voxel_coordinates']
                # 创建稀疏张量（MinkowskiEngine格式）
                # 注意：去掉最后3维（这些是点坐标，已经在稀疏张量的坐标中编码）
                x = ME.SparseTensor(coordinates=voxel_coordinates, features=voxel_features[:, :-3], device=voxel_features.device)
                # 获取体素到分段的映射关系
                voxel2segment = data_dict['voxel2segment']
                # 使用体素编码器处理稀疏张量（通常返回多层特征列表）
                # 注意：voxel_encoder 的返回值应为 list of tensors，代表不同尺度的分段特征
                feat = self.voxel_encoder(x, voxel2segment, max_seg=fts_locs.shape[1])
                
                # 可选：添加几何特征到每个分段
                if self.add_geometry_to_segment:
                    # 遍历每个批次
                    for bid in range(len(voxel2segment)):
                        # 获取该批次的点所属的分段索引
                        sp_idx = voxel2segment[bid]
                        # 获取该批次的所有点坐标
                        all_xyz = data_dict['coordinates'][bid]
                        # 规范化点坐标到单位球体内
                        norm_xyz, _ = scatter_norm(all_xyz, sp_idx)
                        # 通过最大池化操作将规范化坐标聚合到每个分段
                        all_xyz_segment = scatter(self.pts_proj1(norm_xyz), sp_idx, dim=0, reduce='max', dim_size=fts_locs.shape[1])
                        # 将几何特征添加到该批次的所有特征层
                        for i in range(len(feat)):
                            feat[i][bid] = feat[i][bid] + all_xyz_segment
                    
                    # 对所有特征层进行特征投影（融合几何信息）
                    for i in range(len(feat)):        
                        feat[i] = self.feat_proj(torch.cat([feat[i], fts_locs], dim=-1))
                
                # 计算分段的有效掩码（True 表示有效 segment）
                mask = data_dict['seg_pad_masks'].logical_not()
                # 保存体素特征供后续使用（分离计算图以节省内存）
                data_dict['voxel_feat'] = {'feat': feat[-1].detach().cpu(), 'mask': mask.detach().cpu()}
                # 位置信息为分段中心的编码位置
                pos = fts_pos
            else:
                raise NotImplementedError(f"Unknow input type: {input}")
            
            # 将编码后的特征存储到输入字典中
            input_dict[input] = [feat, mask, pos]
        
        # ========== 构建使用的注意力掩码 ==========
        # 离线注意力掩码用于指导掩码训练（当前设置为None）
        offline_attn_masks = None
        
        # ========== 生成用于掩码头的分段特征 ==========
        seg_fts_for_match = []
        for input in self.inputs:
            if input in ['voxel', 'mv']:  # 只选择需要的模态
                feats = copy(input_dict[input][:])
                if isinstance(feats[0], list):
                    # 对于体素模态，使用最后一层特征用于分段匹配
                    assert input == 'voxel'
                    feats[0] = feats[0][-1]
                seg_fts_for_match.append(feats)
        
        # ========== 构建掩码头的偏函数 ==========
        # 使用functools.partial预设一些参数以便后续调用
        if hasattr(self, 'mask_head'):
            mask_head_partial = partial(self.mask_head, query_locs=query_locs, seg_fts_for_match=seg_fts_for_match, seg_masks=data_dict['seg_pad_masks'].logical_not(),
                                        offline_attn_masks=offline_attn_masks, skip_prediction=self.skip_query_encoder_mask_pred)
        else:
            mask_head_partial = None
        
        # ========== 生成空间注意力所需的成对位置关系 ==========
        if self.unified_encoder.spatial_selfattn:
            # 计算查询位置之间的成对相对位置（用于空间自注意力）
            pairwise_locs = calc_pairwise_locs(query_locs[:, :, :3], None, 
                                           pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                           spatial_dim=self.spatial_dim)
        else:
            pairwise_locs = None
        
        # ========== 可选：通过特征初始化查询 ==========
        if self.init_query_by_feat:
            input_query_feat_list = []
            # 从每个模态的分段特征中选择对应查询的特征
            for input in self.inputs:
                if input == 'voxel':
                    # 体素模态使用最后一层特征
                    segment_feature = input_dict['voxel'][0][-1]
                else:
                    # 其他模态直接使用特征
                    segment_feature = input_dict[input][0]
                
                # 获取用于选择查询的分段索引
                query_selection_ids = data_dict['query_selection_ids']
                query_feat_list = []
                
                # 对每个批次提取对应的查询特征
                for bid in range(len(query_selection_ids)):
                    query_feat = segment_feature[bid][query_selection_ids[bid]]
                    query_feat_list.append(query_feat)
                
                # 将特征序列填充到相同长度，并乘以对应模态的权重
                query_feat = pad_sequence(query_feat_list, batch_first=True) * self.input_weights[input]
                input_query_feat_list.append(query_feat)
            
            # 堆叠所有模态的查询特征并求和作为最终查询特征
            # 这里将所有模态的查询特征沿新维度堆叠并按模态权重加权求和
            # 注意：不同模态的 query_feat 通过 pad_sequence 对齐到相同的 S_query 长度
            #       最终 query 形状应与 input_dict['query'][0]（feat）一致
            query = torch.stack(input_query_feat_list, dim=-1).sum(dim=-1)
            # 验证查询特征形状是否正确
            assert query.shape == input_dict['query'][0].shape, f"Query shape {query.shape} does not match input shape {input_dict['query'][0].shape}"
            # 更新输入字典中的查询特征
            input_dict['query'] = (query, input_dict['query'][1], input_dict['query'][2])
        
        # ========== 统一编码 ==========
        # 通过统一编码器融合多个模态的特征并进行交叉注意力
        query, predictions_score, predictions_class, predictions_mask, predictions_box = self.unified_encoder(input_dict, pairwise_locs, mask_head_partial)
        # 保存编码后的查询特征供后续使用
        data_dict['query_feat'] = query
        
        # ========== 任务头处理 ==========
        for head in self.heads:
            if head == 'mask':  # 处理掩码任务
                # 如果在统一编码器中跳过了掩码预测，则在此重新计算
                if self.skip_query_encoder_mask_pred:
                    mask_head_partial = partial(self.mask_head, query_locs=query_locs, seg_fts_for_match=seg_fts_for_match, seg_masks=data_dict['seg_pad_masks'].logical_not(),
                                    offline_attn_masks=offline_attn_masks, skip_prediction=False)
                    # 重置预测列表
                    predictions_score = []
                    predictions_class = []
                    predictions_mask = []
                    predictions_box = []
                
                # 调用掩码头获取最终的预测结果
                pred_scores, pred_logits, pred_masks, pred_boxes, _ = mask_head_partial(query=query)
                # 将预测结果追加到列表中
                predictions_score.append(pred_scores)
                predictions_class.append(pred_logits)
                predictions_mask.append(pred_masks)
                predictions_box.append(pred_boxes)
                
                # 存储最终的预测结果到数据字典
                data_dict['predictions_score'] = predictions_score
                data_dict['predictions_class'] = predictions_class
                data_dict['predictions_mask'] = predictions_mask
                data_dict['predictions_box'] = predictions_box
                continue
            
            elif head == 'openvocab':  # 处理开放词汇任务
                # 使用开放词汇头处理查询特征
                openvocab_query_feat = getattr(self, 'openvocab_head')(query)
                # 存储开放词汇特征到数据字典
                data_dict['openvocab_query_feat'] = openvocab_query_feat
            else:
                raise NotImplementedError(f"Unknow head type: {head}")
       
        return data_dict

    def get_opt_params(self):
        """
        获取优化器参数组。
        
        此方法为模型的不同模块分配可能不同的学习率，以便在优化过程中进行细粒度控制。
        支持为特定模块设置自定义学习率，否则使用默认学习率。
        
        返回:
            optimizer_grouped_parameters: 参数组列表，每个元素为字典形式：
                {
                    'params': [参数列表],
                    'lr': 学习率,
                    'name': 模块名称,
                    'weight_decay': 权重衰减值
                }
        """
        def get_lr(cfg, default_lr):
            """获取学习率，优先使用配置中的值，否则使用默认学习率"""
            return default_lr if cfg is None or cfg.get("lr") is None else cfg.get("lr")

        # 初始化参数组列表
        optimizer_grouped_parameters = []
        
        # 遍历模型的所有模块
        for name, module in self._modules.items():
            # 获取该模块的配置
            module_cfg = self.cfg.model.get(name)
            # 根据配置获取该模块的学习率
            lr = get_lr(module_cfg, self.cfg.solver.lr)
            # 如果学习率与默认学习率不同，打印提示信息
            if lr != self.cfg.solver.lr:
                print(f"Change lr from default {self.cfg.solver.lr} to {lr} for {name} module.")
            # 为该模块的所有参数添加参数组
            # no_decay_param_group 函数会为参数分配权重衰减或不衰减的分组
            optimizer_grouped_parameters += no_decay_param_group(module.named_parameters(), lr, name=name)

        # 验证：提取所有被优化的参数
        optimized_parameters = [p for group in optimizer_grouped_parameters for p in group['params']]
        # 检查所有模型参数是否都被包含在优化器参数组中
        assert len(optimized_parameters) == len(list(self.parameters())), "Some parameters are not optimized!"
        
        return optimizer_grouped_parameters
        