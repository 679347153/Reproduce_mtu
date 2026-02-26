"""
3D导航智能体评估系统
本代码实现了一个基于Habitat仿真环境的3D导航智能体评估系统
主要功能包括：加载数据集、初始化仿真环境、运行导航策略、评估性能指标
"""

from collections import defaultdict
import gzip
import os
import sys
import habitat
from habitat.utils.visualizations import maps
from habitat_sim import Simulator as Sim
import json
import habitat_sim
import numpy as np
from habitat.tasks.nav.nav import TopDownMap
from omegaconf import OmegaConf
import torch
from common.embodied_utils.simulator import HabitatSimulator
from frontier_utils import convert_meters_to_pixel, detect_frontier_waypoints, get_closest_waypoint, get_polar_angle, map_coors_to_pixel, pixel_to_map_coors, reveal_fog_of_war
from sim_utils import get_simulator
import cv2
from data_utils import PQ3DModel
import random

# ==================== 超参数配置 ====================
# 数据集路径
# data_set_path = "/mnt/fillipo/zhuziyu/embodied_bench_data/our-set/goat_full_set.json"
# navigation_data_path = "/mnt/fillipo/zhuziyu/embodied_bench_data/goat/"
# hm3d_data_base_path = "/mnt/fillipo/ML/zhuofan/data/scene_datasets/hm3d/val"
# embodied_scan_dir = "/mnt/fillipo/zhuziyu/embodied_scan"

# 模型路径
# pq3d_stage1_path = "/mnt/fillipo/zhuziyu/embodied_saved_data/saved_models/embodied-pq3d-final/stage1-pretrain-all"
# pq3d_stage2_path = "/mnt/fillipo/zhuziyu/embodied_saved_data/saved_models/embodied-pq3d-final-stage2/stage2-fine-tune-goat-image-rerun"


data_set_path = "/home/ma-user/work/zhangWei/mtu3d/data/embodied_bench_data/our-set/goat_full_set.json"
navigation_data_path = "/home/ma-user/work/zhangWei/mtu3d/data/embodied_bench_data/goat/"
# hm3d_data_base_path = "/mnt/fillipo/ML/zhuofan/data/scene_datasets/hm3d/val"
hm3d_data_base_path = "/home/ma-user/work/zhangWei/mtu3d/data/trans/hm3d"
embodied_scan_dir = "/home/ma-user/work/zhangWei/mtu3d/data/embodied_scan"
pq3d_stage1_path = "/home/ma-user/work/zhangWei/mtu3d/data/trans/MTU3D-c/stage1-pretrain-all"
pq3d_stage2_path = "/home/ma-user/work/zhangWei/mtu3d/data/trans/MTU3D-c/stage2-fine-tune-goat"
output_path = "/home/ma-user/work/zhangWei/mtu3d/record/mtu3d/output_dirs/goat-full-finetune-num-1.json"

# 输出路径
# output_path = "./output_dirs/goat-full-finetune-num-1.json"

# 实验参数
enable_visualization = False  # 是否启用可视化（保存视频和点云）
decision_num_min = 3  # 最小决策次数阈值
visible_radius = 3  # 可见半径（米），用于战争迷雾计算

# ==================== 加载导航数据 ====================
"""
导航数据格式说明：
navigation_data_dict 结构：
{
    'val_seen': {  # 分割集名称
        'scene_id_1': {  # 场景ID
            'episodes': [...],  # 导航episode列表
            'goals_by_category': {...}  # 按类别组织的目标对象
        }
    }
}
"""
navigation_data_dict = {'val_seen': {}, 'val_seen_synonyms': {}, 'val_unseen': {}}
split_list = ['val_seen', 'val_seen_synonyms', 'val_unseen'] 

# 加载训练-验证分割配置
train_val_split = json.load(open(os.path.join(embodied_scan_dir, 'HM3D', 'hm3d_annotated_basis.scene_dataset_config.json')))
# 提取原始扫描ID
raw_scan_ids = set([pa.split('/')[1] for pa in train_val_split['scene_instances']['paths']['.json']])

# 为每个分割集加载导航数据
for split in split_list:
    data_dir = os.path.join(navigation_data_path, split, 'content')
    # 获取数据文件列表（排除隐藏文件）
    file_list = [f for f in os.listdir(data_dir) if f[0] != '.']
    
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        # 使用gzip打开压缩的JSON文件
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)  # 解析JSON数据
            
            # 场景ID处理：简化ID -> 原始ID
            simplified_scan_id = file_name.split('.')[0]
            # 找到对应的原始扫描ID
            raw_scan_id = [pa for pa in raw_scan_ids if simplified_scan_id in pa][0]
            
            # 重组数据结构
            new_data = {}
            new_data['episodes'] = data['episodes']  # 导航episode
            # 处理目标对象，去掉文件名前缀
            new_data['goals_by_category'] = dict([(k.split('glb_')[-1], v) for k, v in data['goals'].items()])
            
            navigation_data_dict[split][raw_scan_id] = new_data

# ==================== 加载图像特征 ====================
"""
图像特征用于基于图像的导航任务
CLIP特征存储在.pth文件中
"""
image_feat_dir = os.path.join('/mnt/fillipo/zhuziyu/embodied_scan_vle_data/', 'goat-clip-feat')
image_feat_dict = {'val_seen': {}, 'val_seen_synonyms': {}, 'val_unseen': {}}

for split in split_list:
    file_list = os.listdir(os.path.join(image_feat_dir, split))
    for f_name in file_list:
        # 加载图像特征，移除文件名扩展名作为键
        image_feat = torch.load(os.path.join(image_feat_dir, split, f_name), map_location='cpu') 
        image_feat_dict[split][f_name.split('.')[0]] = image_feat

# ==================== 加载评估数据集 ====================
"""
数据集包含要评估的episode列表
每个episode包含场景ID、episode索引、目标等信息
"""
data_set = json.load(open(data_set_path, "r"))

# ==================== 结果记录初始化 ====================
"""
结果数据结构：
{
    'split_name': {
        'object': [list_of_results],  # 基于对象的导航结果
        'description': [list_of_results],  # 基于描述的导航结果
        'image': [list_of_results]  # 基于图像的导航结果
    }
}
"""
if os.path.exists(output_path):
    # 如果输出文件已存在，加载已有结果
    result_dict = json.load(open(output_path, "r"))
else:
    # 初始化新的结果字典
    result_dict = {
        'val_seen': {'object': [], 'description': [], 'image': []}, 
        'val_seen_synonyms': {'object': [], 'description': [], 'image': []}, 
        'val_unseen': {'object': [], 'description': [], 'image': []}
    }

# 过滤掉已经处理过的episode
for split in split_list:
    # 提取已有结果中的episode标识（场景ID + episode索引）
    existing_episodes = {
        (result['scan_id'], result['episode_index']) 
        for goal_type in result_dict[split] 
        for result in result_dict[split][goal_type]
    }
    # 过滤掉已处理的episode
    data_set[split] = [
        episode for episode in data_set[split] 
        if (episode['scan_id'], episode['episode_index']) not in existing_episodes
    ]

# ==================== 加载PQ3D模型 ====================
"""
PQ3D模型是一个基于点云的3D导航模型
stage1: 预训练模型
stage2: 在特定任务上微调的模型
"""
pq3d_model = PQ3DModel(pq3d_stage1_path, pq3d_stage2_path, min_decision_num=decision_num_min)

# ==================== 主评估循环 ====================
# 遍历所有分割集
for split in split_list:
    # 遍历当前分割集中的所有episode
    for cur_data in data_set[split]:
        # ---------- 加载当前episode ----------
        scene_id = cur_data['scan_id']
        clean_scene_id = scene_id.split("-")[-1]  # 清理场景ID
        # 构建场景文件路径
        scene_path = os.path.join(hm3d_data_base_path, scene_id, f"{clean_scene_id}.basis.glb")
        episode_index = cur_data['episode_index']
        # 获取当前episode的详细信息
        cur_episode = navigation_data_dict[split][scene_id]['episodes'][episode_index]

        # 重置PQ3D模型状态
        pq3d_model.reset()
        
        # ---------- 初始化导航起点 ----------
        start_position = cur_episode['start_position']
        start_rotation = cur_episode['start_rotation']
        
        # ---------- 初始化仿真器 ----------
        # 加载仿真器配置
        sim_settings = OmegaConf.load('configs/habitat/goat_sim_config.yaml')
        goat_agent_setting = OmegaConf.load('configs/habitat/goat_agent_config.yaml')
        sim_settings['scene'] = scene_path
        
        # 创建Habitat仿真器实例
        abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
        sim = abstract_sim.simulator
        agent = abstract_sim.agent
        
        # 设置智能体初始状态
        agent_state = habitat_sim.AgentState()
        agent_state.position = start_position
        agent_state.rotation = start_rotation
        agent.set_state(agent_state)
        
        # 获取路径查找器
        path_finder = sim.pathfinder
        
        # ---------- 初始化地图相关参数 ----------
        map_resolution = 512  # 地图分辨率
        # 获取场景的俯视图
        top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution, draw_border=False)
        # 初始化战争迷雾掩码（0=未探索，1=已探索）
        fog_of_war_mask = np.zeros_like(top_down_map)
        # 面积阈值（用于边界点检测），9平方米转换为像素
        area_thres_in_pixels = convert_meters_to_pixel(9, map_resolution, sim)
        # 可见距离转换为像素
        visibility_dist_in_pixels = convert_meters_to_pixel(visible_radius, map_resolution, sim)
        
        # ---------- 初始化全局变量 ----------
        decision_num = 0  # 决策计数器
        global_color_list = []  # 用于记录所有颜色帧（可视化用）
        visited_frontier_set = set()  # 已访问的边界点集合
        
        # ---------- 子episode循环 ----------
        # 每个episode可能包含多个子任务
        for sub_episode_index in range(len(cur_episode['tasks'])):
            # 构建当前子任务的目标
            cur_sub_episode = cur_episode['tasks'][sub_episode_index]
            
            # 解析目标信息
            if len(cur_sub_episode) == 3:
                # 格式：目标类别，目标类型，目标对象ID
                goal_category, goal_type, goal_object_id = cur_sub_episode
            else:
                # 格式：目标类别，目标类型，目标对象ID，目标图像ID
                goal_category, goal_type, goal_object_id, goal_image_id = cur_sub_episode
            
            # 根据目标类型设置提示信息和目标列表
            if goal_type == 'object':
                # 基于对象：使用对象类别作为语言提示
                sentence = goal_category
                goals = [g for g in navigation_data_dict[split][scene_id]['goals_by_category'][goal_category]]
                
            elif goal_type == 'description':
                # 基于描述：使用语言描述作为提示
                goals = [g for g in navigation_data_dict[split][scene_id]['goals_by_category'][goal_category] 
                        if g['object_id'] == goal_object_id]
                sentence = goals[0]['lang_desc']  # 获取语言描述
                assert len(goals) == 1
                
            elif goal_type == 'image':
                # 基于图像：使用CLIP图像特征
                sentence = goal_category
                goals = [g for g in navigation_data_dict[split][scene_id]['goals_by_category'][goal_category] 
                        if g['object_id'] == goal_object_id]
                # 获取目标图像特征
                goal_image_feat = image_feat_dict[split][scene_id][int(goal_object_id.split('_')[1])][goal_image_id]
                assert len(goals) == 1
                
            print(f"目标: {sentence} (类型: {goal_type})")
            
            # ---------- 子episode初始化 ----------
            total_steps = 0  # 总步数计数器
            prev_agent_state = agent.get_state()  # 记录初始状态
            sub_episode_start_position = prev_agent_state.position  # 子任务起始位置
            episode_cum_distance = 0  # 累积移动距离
            
            # 用于存储移动过程中的观察数据
            goto_color_list = []  # 移动过程中的颜色图像
            goto_depth_list = []  # 移动过程中的深度图像
            goto_agent_state_list = []  # 移动过程中的智能体状态
            
            # ---------- 导航主循环 ----------
            # 最大步数为500
            while total_steps < 500:
                # ---------- 旋转观察 ----------
                color_list = []
                depth_list = []
                agent_state_list = []
                
                # 对之前的移动观察进行下采样（最多保留6帧）
                if len(goto_color_list) > 6:
                    # 等间隔采样
                    goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), len(goto_color_list) // 6)][:6]
                    goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), len(goto_depth_list) // 6)][:6]
                    goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), len(goto_agent_state_list) // 6)][:6]
                
                # 添加之前的观察
                color_list.extend(goto_color_list)
                depth_list.extend(goto_depth_list)
                agent_state_list.extend(goto_agent_state_list)
                
                # ---------- 执行旋转动作 ----------
                # 向左旋转12次，每次30度，总共360度
                action_list = ['turn_left'] * 12
                for action in action_list:
                    # 执行动作，获取观察
                    observations = sim.step(action=action)
                    # 颜色传感器数据 (h, w, 4)，提取RGB通道
                    color = observations['color_sensor'][:, :, :3]
                    color_list.append(color)
                    global_color_list.append(color)  # 添加到全局列表（用于可视化）
                    
                    # 深度传感器数据
                    depth = observations['depth_sensor'][:, :]
                    depth_list.append(depth)
                    
                    # 获取当前智能体状态
                    agent_state = agent.get_state()
                    agent_state_list.append(agent_state)
                    
                    # 可视化：保存当前帧
                    if enable_visualization:
                        cv2.imwrite('color.png', color)
                    
                    # 更新战争迷雾
                    fog_of_war_mask = reveal_fog_of_war(
                        top_down_map=top_down_map,
                        current_fog_of_war_mask=fog_of_war_mask,
                        current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim),
                        current_angle=get_polar_angle(agent_state),
                        fov=42,  # 视野角度
                        max_line_len=visibility_dist_in_pixels,  # 最大可见距离
                        enable_debug_visualization=enable_visualization
                    )
                    total_steps += 1
                
                # 获取旋转后的智能体状态
                agent_state = agent.get_state()
                
                # ---------- 检测边界点 ----------
                # 转换当前位置为像素坐标
                agent_pixel_pos = map_coors_to_pixel(agent_state.position, top_down_map, sim)
                # 检测边界点（可探索区域边界）
                frontier_waypoints = detect_frontier_waypoints(
                    top_down_map, 
                    fog_of_war_mask, 
                    area_thres_in_pixels, 
                    xy=agent_pixel_pos[::-1],  # 反转坐标顺序
                    enable_visualization=enable_visualization
                )
                
                # 处理边界点
                if len(frontier_waypoints) == 0:
                    frontier_waypoints = []
                else:
                    # 坐标轴转换
                    frontier_waypoints = frontier_waypoints[:, ::-1]
                    # 像素坐标转换为地图坐标
                    frontier_waypoints = pixel_to_map_coors(frontier_waypoints, agent_state.position, top_down_map, sim)
                
                # 过滤已访问的边界点
                frontier_waypoints = [
                    waypoint for waypoint in frontier_waypoints 
                    if tuple(np.round(waypoint, 1)) not in visited_frontier_set
                ] 
                
                # ---------- 决策 ----------
                try:
                    if goal_type == 'image':
                        # 基于图像的导航：传入图像特征
                        target_position, is_final_decision = pq3d_model.decision(
                            color_list, 
                            depth_list, 
                            agent_state_list, 
                            frontier_waypoints, 
                            sentence, 
                            decision_num, 
                            goal_image_feat
                        )
                    else:
                        # 基于对象或描述的导航
                        target_position, is_final_decision = pq3d_model.decision(
                            color_list, 
                            depth_list, 
                            agent_state_list, 
                            frontier_waypoints, 
                            sentence, 
                            decision_num
                        )
                except Exception as e:
                    print(f"决策错误, episode_id: {cur_episode['episode_id']}, 场景: {scene_id}, 错误: {e}")
                    sys.exit(1)
                    break
                
                decision_num += 1
                
                # 如果不是最终决策，将目标点标记为已访问
                if not is_final_decision:
                    visited_frontier_set.add(tuple(np.round(target_position, 1)))
                
                # ---------- 导航到目标点 ----------
                # 获取智能体所在的导航网格岛屿
                agent_island = path_finder.get_island(agent_state.position)
                # 将目标点吸附到导航网格上
                target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
                
                # 创建贪婪地理距离跟随器
                follower = habitat_sim.GreedyGeodesicFollower(
                    path_finder, 
                    agent, 
                    forward_key="move_forward", 
                    left_key="turn_left", 
                    right_key="turn_right"
                )
                
                # 计算路径
                try:
                    action_list = follower.find_path(target_on_navmesh)
                except:
                    # 路径计算失败处理
                    if not path_finder.is_navigable(target_on_navmesh):
                        print("目标点不可导航")
                    if not path_finder.is_navigable(agent_state.position):
                        print("智能体当前位置不可导航")
                    
                    # 尝试计算最短路径
                    path = habitat_sim.ShortestPath()
                    path.requested_start = agent_state.position
                    path.requested_end = target_on_navmesh
                    if sim.pathfinder.find_path(path):
                        print(f"测地线距离: {path.geodesic_distance}")
                    else:
                        print("无法找到路径")
                    action_list = []
                    break
                
                # 重置移动观察列表
                goto_color_list = []
                goto_depth_list = []
                goto_agent_state_list = []
                
                # 执行移动动作
                for action in action_list:
                    if action:
                        # 执行动作
                        observations = sim.step(action=action)
                        # 记录全局颜色帧
                        global_color_list.append(observations['color_sensor'][:, :, :3])
                        
                        # 更新智能体状态
                        agent_state = agent.get_state()
                        
                        # 记录当前观察
                        color = observations['color_sensor'][:, :, :3]
                        goto_color_list.append(color)
                        depth = observations['depth_sensor'][:, :]
                        goto_depth_list.append(depth)
                        goto_agent_state_list.append(agent_state)
                        
                        # 更新战争迷雾
                        fog_of_war_mask = reveal_fog_of_war(
                            top_down_map=top_down_map,
                            current_fog_of_war_mask=fog_of_war_mask,
                            current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim),
                            current_angle=get_polar_angle(agent_state),
                            fov=42,
                            max_line_len=visibility_dist_in_pixels,
                            enable_debug_visualization=enable_visualization
                        )
                        
                        total_steps += 1
                        # 计算移动距离
                        episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                        prev_agent_state = agent_state
                
                # 如果是最终决策，跳出循环
                if is_final_decision:
                    break
                    
            # ---------- 可视化处理 ----------
            if enable_visualization:
                # 保存导航过程为视频
                height, width, layers = global_color_list[0].shape
                video = cv2.VideoWriter(f'video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 2, (width, height))
                for color_frame in global_color_list:
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                    video.write(color_frame)
                video.release()
                
                # 保存彩色点云
                pq3d_model.representation_manager.save_colored_point_cloud()
            
            # ---------- 性能指标计算 ----------
            agent_state = agent.get_state()
            
            # 提取所有目标的可视点位置
            view_points = [
                view_point["agent_state"]["position"]
                for goal in goals
                for view_point in goal["view_points"]
            ]
            
            # 计算起始点到目标的测地线距离
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = sub_episode_start_position
            path.requested_ends = view_points
            if path_finder.find_path(path):
                start_end_geo_distance = path.geodesic_distance
            else:
                print('目标不可导航')
                start_end_geo_distance = np.inf
            
            # 计算智能体当前位置到目标的测地线距离
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = agent_state.position
            path.requested_ends = view_points
            if path_finder.find_path(path):
                agent_end_geo_distance = path.geodesic_distance
            else:
                agent_end_geo_distance = np.inf
            
            # 计算成功率和SPL（成功加权路径长度）
            if start_end_geo_distance == np.inf:
                # 如果起始点不可达，认为是特殊情况
                sr = 1
                spl = 1
            elif agent_end_geo_distance == np.inf:
                # 如果智能体位置不可达目标
                sr = 0
                spl = 0
            else:
                # 成功条件：距离目标小于0.25米
                sr = agent_end_geo_distance <= 0.25
                # SPL公式：成功 * (最短路径长度 / max(最短路径长度, 实际路径长度))
                spl = sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)
            
            # 记录结果
            result_dict[split][goal_type].append({
                'scan_id': scene_id,
                'episode_index': episode_index,
                'sub_episode_index': sub_episode_index,
                'sr': sr,  # 成功率
                'spl': spl,  # 加权路径长度
                'object_category': goal_category
            })
            
            print(f"SR: {sr}, SPL: {spl}, 起始位置: {start_position}, 当前位置: {agent_state.position}, "
                  f"目标位置: {[g['position'] for g in goals]}, 对象类别: {goal_category}, "
                  f"决策次数: {decision_num}, 目标类型: {goal_type}")
        
        # 保存结果到文件
        with open(output_path, "w") as f:
            json.dump(result_dict, f)

# ==================== 结果统计和输出 ====================
# 按分割集和目标类型统计平均性能
for split in split_list:
    for goal_type in ['object', 'description', 'image']:
        total_sr = 0
        total_spl = 0
        # 按类别统计
        category_sr_spl = defaultdict(lambda: {'sr': 0, 'spl': 0, 'count': 0})
        count = 0
        
        for result in result_dict[split][goal_type]:
            total_sr += result['sr']
            total_spl += result['spl']
            category = result['object_category']
            category_sr_spl[category]['sr'] += result['sr']
            category_sr_spl[category]['spl'] += result['spl']
            category_sr_spl[category]['count'] += 1
            count += 1

        # 计算平均值
        avg_sr = total_sr / count if count > 0 else 0
        avg_spl = total_spl / count if count > 0 else 0
        
        print(f"分割集: {split}, 目标类型: {goal_type}, 平均SR: {avg_sr:.4f}, 平均SPL: {avg_spl:.4f}")

        # 输出每个类别的详细结果（注释掉以减少输出）
        # for category, metrics in category_sr_spl.items():
        #     avg_category_sr = metrics['sr'] / metrics['count'] if metrics['count'] > 0 else 0
        #     avg_category_spl = metrics['spl'] / metrics['count'] if metrics['count'] > 0 else 0
        #     print(f"类别: {category}, 平均SR: {avg_category_sr}, 平均SPL: {avg_category_spl}")

# 计算并输出每个分割集的总体性能
for split in split_list:
    total_sr = 0
    total_spl = 0
    count = 0
    
    for goal_type in ['object', 'description', 'image']:
        for result in result_dict[split][goal_type]:
            total_sr += result['sr']
            total_spl += result['spl']
            count += 1

    avg_sr = total_sr / count if count > 0 else 0
    avg_spl = total_spl / count if count > 0 else 0
    
    print(f"分割集: {split}, 总体平均SR: {avg_sr:.4f}, 总体平均SPL: {avg_spl:.4f}")