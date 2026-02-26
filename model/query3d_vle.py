"""
================================================================================
                         Query3DVLE 模型文件
================================================================================

文件概述:
    本文件实现 `Query3DVLE` 模型，用于在 VLE（视觉-语言-探索）场景中，
    基于查询的三维目标检测 / 分类 / 判定任务。

核心架构:
    - prompt_encoder: 多种提示（文本/位置/图像）的编码器接口
    - forward: 构建查询、编码多模态输入、调用统一编码器并执行任务头
    - get_opt_params: 为优化器返回分组参数

主要功能:
    1. 支持多种提示类型（文本、位置信息、图像）
    2. 对分段（segment）级别的特征进行编码与位置嵌入
    3. 将查询与分段特征输入统一编码器（Transformer）进行交互
    4. 提供不同任务头接口（接地、查询分类、决策等）

输入/输出简述:
    输入: data_dict 包含 query_locs, seg_center, 不同模态的分段特征等
    输出: 在 data_dict 中附加各任务头的 logits/labels 等结果

================================================================================
"""

# 标准库
from copy import copy  # 浅拷贝列表/对象
from functools import partial  # 用于创建带固定参数的偏函数

# 第三方库
import torch
import torch.nn as nn
import MinkowskiEngine as ME  # 稀疏张量处理（体素）

# 项目内工具/数据类型
from data.data_utils import pad_sequence  # 可变长度序列填充
from data.datasets.constant import PromptType  # 提示类型常数

from modules.build import build_module_by_name  # 根据配置构建模块
from modules.utils import calc_pairwise_locs  # 计算成对位置关系（用于空间自注意力）
from model.build import MODEL_REGISTRY, BaseModel  # 模型注册与基类
from optim.utils import no_decay_param_group  # 优化器参数分组工具
from model.mask3d import CoordinateEncoder  # 统一使用的坐标编码器
        
@MODEL_REGISTRY.register()
class Query3DVLE(BaseModel):
    def __init__(self, cfg):
        """
        初始化 Query3DVLE 模型。

        参数:
            cfg: 配置对象，包含模型结构与超参
        """
        super().__init__(cfg)
        # ========== 配置记录 ==========
        self.cfg = cfg
        # memories 存放可用输入模态（例如 'prompt','mv','vocab'）
        self.memories = cfg.model.memories
        # 任务头列表（例如 ['ground','query_cls','decision']）
        self.heads = cfg.model.heads
        # 将 memories 复制为实际输入列表
        self.inputs = self.memories[:]
        # 空间相关配置
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim
        # Transformer 多头注意力头数
        self.num_heads = self.cfg.model.unified_encoder.args.num_attention_heads

        # ========== 支持的提示类型 ==========
        self.prompt_types = ['txt', 'loc', 'image']

        # ========== 构建各模态编码器 ==========
        # 对于 prompt 输入，需要分别为每种提示类型构建编码器
        for input in self.inputs:
            if input == 'prompt':
                for prompt_type in self.prompt_types:
                    # 位置型提示（loc）单独处理，当前跳过自动构建（如需可开启）
                    if prompt_type == 'loc':
                        continue
                    encoder = prompt_type + '_encoder'
                    setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
            else:
                encoder = input + '_encoder'
                setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))

        # ========== 位置/盒子 编码器 ==========
        dim_loc = self.cfg.model.obj_loc.dim_loc
        hidden_size = self.cfg.model.hidden_size
        self.dim_loc = dim_loc
        self.hidden_size = hidden_size
        # 用于对中心点编码（简单线性 + LayerNorm）
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        # 用于对box尺寸或其他3维信息编码
        self.box_encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # ========== 统一编码器与任务头 ==========
        # 统一编码器（Transformer 风格）用于融合 query 与各模态特征
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        # 根据配置构建各任务头
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))

        # ========== 额外嵌入 ==========
        # frontier_embedding: 为前沿（frontier）或真实目标标识提供嵌入
        self.frontier_embedding = nn.Embedding(2, self.hidden_size)
        
    def prompt_encoder(self, data_dict):
        """
        对输入提示（prompt）进行编码，支持三类提示：文本、位置、图像。

        参数:
            data_dict: 包含 'prompt', 'prompt_pad_masks', 'prompt_type' 等字段

        返回:
            prompt_feat: 编码后的特征，形状与输入提示匹配
            prompt_valid_mask: 反向掩码（True 表示有效元素）
        """
        prompt = data_dict['prompt']
        prompt_pad_masks = data_dict['prompt_pad_masks']
        prompt_type = data_dict['prompt_type']
        # 初始化输出特征张量
        prompt_feat = torch.zeros(prompt_pad_masks.shape + (self.hidden_size,), device=prompt_pad_masks.device)

        # 针对每种提示类型分别编码并回填到原始索引位置
        for type in self.prompt_types:
            # 找到当前 batch 中属于该类型的样本索引
            idx = prompt_type == getattr(PromptType, type.upper())
            if idx.sum() == 0:
                continue
            # 收集属于该类型的输入（变长情况以列表保存）
            input = []
            for i in range(len(prompt)):
                if idx[i]:
                    input.append(prompt[i])
            mask = prompt_pad_masks[idx]

            # 根据类型调用不同编码器
            if type == 'txt':
                input = pad_sequence(input, pad=0)
                encoder = self.txt_encoder
                feat = encoder(input.long(), mask)
            elif type == 'loc':
                # loc 提示包含位置信息（中心/box）
                loc_prompts = input[:, :self.dim_loc]
                if self.dim_loc > 3:
                    # 前3维为中心，后3维为box尺寸或offset
                    feat = self.coord_encoder(loc_prompts[:, :3]).unsqueeze(1) + self.box_encoder(loc_prompts[:, 3:6]).unsqueeze(1)
                else:
                    # 仅中心点
                    feat = self.coord_encoder(loc_prompts[:, :3].unsqueeze(1), input_range=[data_dict['coord_min'][idx], data_dict['coord_max'][idx]])
                # 对位置提示，只有第1列为真实内容，其它位置设置为无效
                mask[:, 1:] = False
            elif type == 'image':
                img_prompts = torch.stack(input).unsqueeze(1)
                feat = self.image_encoder(img_prompts)
                mask[:, 1:] = False
            else:
                raise NotImplementedError(f'{type} is not implemented')

            # 将编码后的特征放回原始 batch 的对应位置
            prompt_feat[idx] = feat
            prompt_pad_masks[idx] = mask

        # 返回特征与反向掩码（True 表示有效）
        return prompt_feat, prompt_pad_masks.logical_not()
        
    def forward(self, data_dict):
        """
        前向传播主流程。

        步骤概要:
            1. 构建 query（包含位置编码、前沿嵌入等）
            2. 对各输入模态进行编码 (prompt/mv/vocab)
            3. 计算成对位置关系（可选，用于空间自注意力）
            4. 调用统一编码器进行交互
            5. 调用各任务头并将结果写回 data_dict
        """
        input_dict = {}

        # ========== 构建查询（Query） ==========
        mask = data_dict['query_pad_masks'].logical_not()  # True 表示有效位置
        query_locs = data_dict['query_locs'][:, :, :self.dim_loc]
        # 位置由中心点编码与box编码相加得到
        query_pos  = self.coord_encoder(query_locs[:, :, :3]) + self.box_encoder(query_locs[:, :, 3:6])
        # 初始查询特征为零，再加上 frontier_emb 表示是否为 frontier/真实目标
        feat = torch.zeros_like(query_pos)
        real_obj_pad_masks = data_dict['real_obj_pad_masks']
        frontier_emb = self.frontier_embedding(real_obj_pad_masks.long())
        feat += frontier_emb
        pos = query_pos
        input_dict['query'] = (feat, mask, pos)

        # ========== 编码分段/模态特征 ==========
        # fts_locs: 每个分段的位置信息
        fts_locs = data_dict['seg_center']
        fts_pos = self.coord_encoder(fts_locs[:, :, :3]) + self.box_encoder(fts_locs[:, :,  3:6])
        for input in self.inputs:
            feat, mask, pos = None, None, None
            if input == 'prompt':
                feat, mask = self.prompt_encoder(data_dict)
            elif input == 'mv':
                feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                mask = data_dict['mv_seg_pad_masks'].logical_not()
                pos = fts_pos
            elif input == 'vocab':
                feat = self.vocab_encoder(data_dict['vocab_seg_fts'])
                mask = data_dict['vocab_seg_pad_masks'].logical_not()
                pos = fts_pos
            else:
                raise NotImplementedError(f'{input} is not implemented')
            input_dict[input] = [feat, mask, pos]

        # 当前模型不使用 mask_head_partial（保留接口）
        mask_head_partial = None

        # ========== 计算空间成对位置（可选） ==========
        if self.unified_encoder.spatial_selfattn:
            pairwise_locs = calc_pairwise_locs(query_locs[:, :, :3], None, 
                                           pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                           spatial_dim=self.spatial_dim)
        else:
            pairwise_locs = None

        # ========== 统一编码（Transformer） ==========
        query, predictions_score, predictions_class, predictions_mask, predictions_box = self.unified_encoder(input_dict, pairwise_locs, mask_head_partial)

        # ========== 任务头处理 ==========
        for head in self.heads:
            if head == 'ground':
                # 接地(head)：使用 query 来预测目标 id
                inputs = [query, data_dict['query_pad_masks']]
                label = data_dict["tgt_object_id"]
                logits = getattr(self, head + '_head')(*inputs)
                data_dict[head + '_logits'] = logits
                data_dict['og3d_logits'] = logits
                data_dict[head + '_label'] = label
            elif head == 'query_cls':
                # 查询分类(head)
                label = data_dict["obj_labels"]
                logits = getattr(self, head + '_head')(query)
                data_dict[head + '_logits'] = logits
                data_dict[head + '_label'] = label
            elif head == 'decision':
                # 决策(head)，可基于 query 与掩码做判定
                label = data_dict['decision_label']
                logits = getattr(self, head + '_head')(query, data_dict['query_pad_masks'])
                data_dict[head + '_logits'] = logits
                data_dict[head + '_label'] = label
            else:
                raise NotImplementedError(f"Unknow head type: {head}")
       
        return data_dict

    def get_opt_params(self):
        """
        返回优化器参数分组，支持为不同模块指定不同学习率。
        """
        def get_lr(cfg, default_lr):
            return default_lr if cfg is None or cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        for name, module in self._modules.items():
            module_cfg = self.cfg.model.get(name)
            lr = get_lr(module_cfg, self.cfg.solver.lr)
            if lr != self.cfg.solver.lr:
                print(f"Change lr from default {self.cfg.solver.lr} to {lr} for {name} module.")
            optimizer_grouped_parameters += no_decay_param_group(module.named_parameters(), lr, name=name)

        optimized_parameters = [p for group in optimizer_grouped_parameters for p in group['params']]
        assert len(optimized_parameters) == len(list(self.parameters())), "Some parameters are not optimized!"
        return optimizer_grouped_parameters
        