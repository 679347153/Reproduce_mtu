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
from common.embodied_utils.simulator import HabitatSimulator
from frontier_utils import convert_meters_to_pixel, detect_frontier_waypoints, get_closest_waypoint, get_polar_angle, map_coors_to_pixel, pixel_to_map_coors, reveal_fog_of_war
from sim_utils import get_simulator
import cv2
from data_utils import PQ3DModel
import random

# hyperparameter
data_set_path = "/home/ma-user/work/zhangWei/mtu3d/data/embodied_bench_data/our-set/sg3d_full_set.json"
navigation_data_path = "/home/ma-user/work/zhangWei/mtu3d/data/embodied_bench_data/sg3d/"
hm3d_data_base_path = "/home/ma-user/work/zhangWei/mtu3d/data/trans/hm3d"
embodied_scan_dir = "/home/ma-user/work/zhangWei/mtu3d/data/embodied_scan"
pq3d_stage1_path = "/home/ma-user/work/zhangWei/mtu3d/data/trans/MTU3D-c/stage1-pretrain-all"
pq3d_stage2_path = "/home/ma-user/work/zhangWei/mtu3d/data/trans/MTU3D-c/stage2-fine-tune-sg3d"
output_path = "/home/ma-user/work/zhangWei/mtu3d/record/mtu3d/output_dirs/sg3d-full-finetune-num-1.json"
enable_visualization = False
decision_num_min = 3
visible_radius = 3

# load navigation data
navigation_data_dict = {'val': {}}
split_list = ['val']
train_val_split = json.load(open(os.path.join(embodied_scan_dir, 'HM3D', 'hm3d_annotated_basis.scene_dataset_config.json')))
raw_scan_ids = set([pa.split('/')[1] for pa in train_val_split['scene_instances']['paths']['.json']])
for split in split_list:
    data_dir = os.path.join(navigation_data_path, split, 'content')
    file_list = [f for f in os.listdir(data_dir) if f[0] != '.']
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)  # 解析 JSON
            # key of data is episodes, goals
            simplified_scan_id = file_name.split('.')[0]
            raw_scan_id = [pa for pa in raw_scan_ids if simplified_scan_id in pa][0]
            new_data = {}
            new_data['episodes'] = data['episodes']
            new_data['goals_by_category'] = dict([(k.split('glb_')[-1], v) for k, v in data['goals'].items()])
            navigation_data_dict[split][raw_scan_id] = new_data

# load data set
data_set = json.load(open(data_set_path, "r"))
            
# record result
if os.path.exists(output_path):
    result_dict = json.load(open(output_path, "r"))
else:
    result_dict = {'val': []}
# filter out data in data_set which is already in result_dict
for split in split_list:
    existing_episodes = {(result['scan_id'], result['episode_index']) for result in result_dict[split]}
    data_set[split] = [episode for episode in data_set[split] if (episode['scan_id'], episode['episode_index']) not in existing_episodes]
    
# load pq3d model
pq3d_model = PQ3DModel(pq3d_stage1_path, pq3d_stage2_path, min_decision_num=decision_num_min)

for split in split_list:
    for cur_data in data_set[split]:
        # load cur episode
        scene_id = cur_data['scan_id']
        clean_scene_id = scene_id.split("-")[-1]
        scene_path = os.path.join(hm3d_data_base_path, scene_id, f"{clean_scene_id}.basis.glb")
        episode_index = cur_data['episode_index']
        cur_episode = navigation_data_dict[split][scene_id]['episodes'][episode_index]

        # reset pq3d
        pq3d_model.reset()
        
        # load target
        start_position = cur_episode['start_position']
        start_rotation = cur_episode['start_rotation']
        
        # get simulator
        sim_settings = OmegaConf.load('configs/habitat/goat_sim_config.yaml')
        goat_agent_setting = OmegaConf.load('configs/habitat/goat_agent_config.yaml')
        sim_settings['scene'] = scene_path
        abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
        sim = abstract_sim.simulator
        agent = abstract_sim.agent
        agent_state = habitat_sim.AgentState()
        agent_state.position = start_position
        agent_state.rotation = start_rotation
        agent.set_state(agent_state)
        path_finder = sim.pathfinder
        
        # get fronier param
        map_resolution = 512
        top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution,  draw_border=False)
        fog_of_war_mask = np.zeros_like(top_down_map)
        area_thres_in_pixels =  convert_meters_to_pixel(9, map_resolution, sim)
        visibility_dist_in_pixels = convert_meters_to_pixel(visible_radius, map_resolution, sim)
        
        # episode global parameter
        decision_num = 0
        global_color_list = []
        visited_frontier_set = set()
        sentence = ""
        # start for loop
        for sub_episode_index in range(len(cur_episode['tasks'])):
            # build goal
            goal_category, goal_type, goal_object_id, sub_sentence = cur_episode['tasks'][sub_episode_index]
            sentence += sub_sentence
            print(sentence)
            assert goal_type == 'description'
            goals = [g for g in navigation_data_dict[split][scene_id]['goals_by_category'][goal_category] if g['object_id'] == goal_object_id]
            assert len(goals) == 1
            # start decision
            # sub_episdoe global parameter 
            total_steps = 0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            episode_cum_distance = 0
            # loop parameter
            goto_color_list = []
            goto_depth_list = []
            goto_agent_state_list = []
            # visited frontier
            while total_steps < 500:
                # spin around
                color_list = []
                depth_list = []
                agent_state_list = []
                # subsample at most 6 frames from goto list, use interval sample
                if len(goto_color_list) > 6:
                    goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), len(goto_color_list) // 6)][:6]
                    goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), len(goto_depth_list) // 6)][:6]
                    goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), len(goto_agent_state_list) // 6)][:6]
                color_list.extend(goto_color_list)
                depth_list.extend(goto_depth_list)
                agent_state_list.extend(goto_agent_state_list)
                # spin
                action_list = ['turn_left'] * 12
                for action in action_list:
                    obervations = sim.step(action=action)
                    color = obervations['color_sensor'][:, :, :3] # (h,w,4) 0-255
                    color_list.append(color)
                    global_color_list.append(color)
                    depth = obervations['depth_sensor'][:, :] # (h,w) float
                    depth_list.append(depth)
                    agent_state = agent.get_state()
                    agent_state_list.append(agent_state)
                    if enable_visualization:
                        # Save the current color image to color.png
                        cv2.imwrite('color.png', color)
                    fog_of_war_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=42, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                    total_steps += 1
                agent_state = agent.get_state()
                # compute frontier
                frontier_waypoints = detect_frontier_waypoints(top_down_map, fog_of_war_mask, area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=enable_visualization)
                if len(frontier_waypoints) == 0:
                    frontier_waypoints = []
                else:
                    frontier_waypoints = frontier_waypoints[:, ::-1]
                    frontier_waypoints = pixel_to_map_coors(frontier_waypoints, agent_state.position, top_down_map, sim)
                # filter out visited frontier
                frontier_waypoints = [waypoint for waypoint in frontier_waypoints if tuple(np.round(waypoint, 1)) not in visited_frontier_set]
                # decision
                try:
                    target_position, is_final_decision = pq3d_model.decision(color_list, depth_list, agent_state_list, frontier_waypoints, sentence, decision_num)
                except Exception as e:
                    print(f"Error in decision making, episode_id: {cur_episode['episode_id']}, scene_id: {scene_id}, {e}")
                    sys.exit(1)
                    break
                decision_num += 1
                # add frontier to visited frontier
                if not is_final_decision:
                    visited_frontier_set.add(tuple(np.round(target_position, 1)))
                # goto
                agent_island = path_finder.get_island(agent_state.position)
                target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
                follower = habitat_sim.GreedyGeodesicFollower(path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
                try:
                    action_list = follower.find_path(target_on_navmesh)
                except:
                    if not path_finder.is_navigable(target_on_navmesh):
                        print("Target is not navigable")
                    if not path_finder.is_navigable(agent_state.position):
                        print("Agent is not navigable")
                    path = habitat_sim.ShortestPath()
                    path.requested_start = agent_state.position
                    path.requested_end = target_on_navmesh
                    if sim.pathfinder.find_path(path):
                        print(f"geodesic_distance: {path.geodesic_distance}")
                    else:
                        print("cannt find path")
                    action_list = []
                    break
                goto_color_list = []
                goto_depth_list = []
                goto_agent_state_list = []
                for action in action_list:
                    if action:
                        obervations = sim.step(action=action)
                        global_color_list.append(obervations['color_sensor'][:, :, :3])
                        agent_state = agent.get_state()
                        color = obervations['color_sensor'][:, :, :3] # (h,w,4) 0-255
                        goto_color_list.append(color)
                        depth = obervations['depth_sensor'][:, :] # (h,w) float
                        goto_depth_list.append(depth)
                        goto_agent_state_list.append(agent_state)
                        fog_of_war_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=42, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                        total_steps += 1
                        episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                        prev_agent_state = agent_state
                # break on final decision
                if is_final_decision:
                    break    
            # Save color_list to video
            if enable_visualization:
                height, width, layers = global_color_list[0].shape
                video = cv2.VideoWriter(f'video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 2, (width, height))
                for color_frame in global_color_list:
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                    video.write(color_frame)
                video.release()
                pq3d_model.representation_manager.save_colored_point_cloud()
            # compute metric
            agent_state = agent.get_state()
            view_points = [
                        view_point["agent_state"]["position"]
                        for goal in goals
                        for view_point in goal["view_points"]
            ]
            # computer start end geodesic distance
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = sub_episode_start_position
            path.requested_ends = view_points
            if path_finder.find_path(path):
                start_end_geo_distance = path.geodesic_distance
            else:
                print('goal is not navigatable')
                start_end_geo_distance = np.inf
            # compute agent current distance
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = agent_state.position
            path.requested_ends = view_points
            if path_finder.find_path(path):
                agent_end_geo_distance = path.geodesic_distance
            else:
                agent_end_geo_distance = np.inf
            # compute success rate
            if start_end_geo_distance == np.inf:
                sr = 1
                spl = 1
            elif agent_end_geo_distance == np.inf:
                sr = 0
                spl = 0
            else:
                sr = agent_end_geo_distance <= 0.25
                spl = sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)
            result_dict[split].append({'scan_id': scene_id, 'episode_index': episode_index, 'sub_episode_index': sub_episode_index, 'sr': sr, 'spl': spl, 'object_category': goal_category})
            print(f"SR: {sr}, SPL: {spl}, Agent start position: {start_position}, Agent position: {agent_state.position}, Goal positions: {[g['position'] for g in goals]}, Object category: {goal_category}, Decision number: {decision_num}")
        with open(output_path, 'w') as f:
            json.dump(result_dict, f)

# Calculate and print average SPL and SR for each split
for split in split_list:
    total_sr = 0
    total_spl = 0
    category_sr_spl = defaultdict(lambda: {'sr': 0, 'spl': 0, 'count': 0})
    for result in result_dict[split]:
        total_sr += result['sr']
        total_spl += result['spl']
        category = result['object_category']
        category_sr_spl[category]['sr'] += result['sr']
        category_sr_spl[category]['spl'] += result['spl']
        category_sr_spl[category]['count'] += 1

    avg_sr = total_sr / len(result_dict[split])
    avg_spl = total_spl / len(result_dict[split])
    print(f"Split: {split}, Average SR: {avg_sr}, Average SPL: {avg_spl}")

    for category, metrics in category_sr_spl.items():
        avg_category_sr = metrics['sr'] / metrics['count']
        avg_category_spl = metrics['spl'] / metrics['count']
        print(f"Category: {category}, Average SR: {avg_category_sr}, Average SPL: {avg_category_spl}")
        
    # Calculate SR averaged by episode_index
    episode_success_count = defaultdict(int)
    episode_total_count = defaultdict(int)

    for result in result_dict[split]:
        episode_key = (result['scan_id'], result['episode_index'])
        episode_total_count[episode_key] += 1
        if result['sr'] == 1:
            episode_success_count[episode_key] += 1

    total_episode_sr = 0
    for episode_key, total_count in episode_total_count.items():
        if episode_success_count[episode_key] == total_count:
            total_episode_sr += 1

    avg_episode_sr = total_episode_sr / len(episode_total_count)
    print(f"Split: {split}, Average Task SR: {avg_episode_sr}")
        