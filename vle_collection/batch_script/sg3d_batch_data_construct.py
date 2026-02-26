import sys
sys.path.append("../vle_collection")
import json
import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf
from simulator import Simulator
import os
from frontier_utils import *
from path_utils import path_dist_cost
import argparse
from torch import multiprocessing as mp
import datetime
import gzip
import torch

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

def preprocess(nav_data : dict):
    episodes = nav_data['episodes']
    goals = nav_data['goals']
    goals_by_category = dict([(k.split("glb_")[-1], goals[k])for k in goals.keys()])
    for key in goals_by_category.keys():
        goals_by_category[key] = {k['object_id']: k for k in goals_by_category[key]}
    return episodes, goals_by_category

def handle_scene(
    scene_id, 
    args,
    ):
    scene_tmp = scene_id.split('-')[1]

    # if os.path.exists(f"{args.output_dir}/{scene_id}.json"):
    #     print(f"{scene_id} already generated.")
    #     return
    
    with gzip.open(f"data/sg3d/{args.split}/content/{scene_tmp}.json.gz", "rt", encoding="utf-8") as f:
        nav_data = json.load(f)
    
    episodes, goals_by_category = preprocess(nav_data)

    goat_sim_settings = OmegaConf.load('config/goat_frontier_sim_config.yaml') #default simulator settings

    scene_tmp = scene_id.split('-')[1]

    #fill in "scene_dataset_config_file" and "scene" ind sim_settings
    goat_sim_settings["scene"] = f"{args.input_scene_dir}/{args.split}/{scene_id}/{scene_tmp}.basis.glb"

    goat_agent_settings = OmegaConf.load('config/goat_frontier_agent_config.yaml')
    agents_settings = [goat_agent_settings]
    sim = Simulator(sim_settings=goat_sim_settings, agents_settings=agents_settings)
    navmesh_setting = habitat_sim.NavMeshSettings()
    navmesh_setting.set_defaults()
    navmesh_setting.agent_height = goat_agent_settings['agent_height']
    navmesh_setting.agent_radius = goat_agent_settings['agent_radius']
    navmesh_setting.agent_max_climb = goat_sim_settings['agent_max_climb']
    navmesh_setting.cell_height = goat_sim_settings['cell_height']
    sim._simulator.recompute_navmesh(sim._simulator.pathfinder, navmesh_setting)
    ### build navmesh

    hfov = goat_agent_settings['hfov']
    vis_frontier = False
    decision_dict = {}
    exception_dict = {}
    start_time = datetime.datetime.now()
    print(f"Start Time of {scene_id}", start_time)
    for epi_id in range(len(episodes)):
        try:
            random_ = np.random.rand()
            if random_ < 0.2:
                strategy = 'best'
            elif 0.2 <= random_ < 0.4:
                strategy = 'closest'
            else:
                strategy = 'mixed'
            decision = save_trajectory(
                epi_id,
                sim,
                episodes,
                goals_by_category,
                hfov,
                vis_frontier,
                strategy,
            )
            decision_dict[epi_id] = decision
        except Exception as e:
            exception_dict[epi_id] = str(e)
            continue

    sim.__del__()
    if not os.path.exists(f"{args.output_dir}/{args.split}"):
        os.makedirs(f"{args.output_dir}/{args.split}")
    with open(f"{args.output_dir}/{args.split}/{scene_id}.json", "w") as f:
        json.dump(decision_dict, f, indent=4)
    os.makedirs(f"tmp/exception_sg3d/{args.split}", exist_ok=True)
    if len(exception_dict) > 0:
        with open(f"tmp/exception_sg3d/{args.split}/{scene_id}.json", "w") as f:
            json.dump(exception_dict, f, indent=4)
    end_time = datetime.datetime.now()
    print(f"Scene {scene_id} done at", end_time, "Total time", end_time - start_time)

    del decision_dict
    del exception_dict
    del episodes
    del goals_by_category
    del sim
    torch.cuda.empty_cache()

def save_trajectory(
    epi_id,
    sim,
    episodes,
    goals_by_category,
    hfov,
    vis_frontier = False,
    start_strategy = 'best',
):
    cur_episode = episodes[epi_id]
    start_position = cur_episode['start_position']
    start_rotation = cur_episode['start_rotation']

    sim.initialize_agent(agent_id=0, position=start_position, rotation=start_rotation)

    cur_island = sim.get_island(start_position)

    map_resolution = 512
    top_down_map = maps.get_topdown_map_from_sim(sim._simulator, map_resolution=map_resolution, draw_border=False) 
    fog_of_war_mask = np.zeros_like(top_down_map) # explored map
    area_thres_in_pixels =  convert_meters_to_pixel(9, map_resolution, sim._simulator)
    visibility_dist_in_pixels = convert_meters_to_pixel(2, map_resolution, sim._simulator)
    visited_frontiers = [] # visited frontier list
    visible_ids = set() # visible instance ids
    ins_list = []

    strategy = start_strategy

    output_decision_list = []
    prev_prompt = ""
    for task in cur_episode['tasks']:
        this_decision, fog_of_war_mask, prev_prompt = sub_trajectory(
            sim,
            hfov,
            task,
            goals_by_category,
            top_down_map,
            fog_of_war_mask,
            area_thres_in_pixels,
            visibility_dist_in_pixels,
            visited_frontiers,
            visible_ids,
            ins_list,
            vis_frontier = vis_frontier,
            strategy = strategy,
            prev_prompt = prev_prompt,
        )
        output_decision_list.append(this_decision)
        if strategy == 'best':
            continue
        if len(this_decision) > 10:
            if strategy == 'closest':
                strategy = 'mixed'
            elif strategy == 'mixed':
                strategy = 'best'
            

    return output_decision_list

def sub_trajectory(
    sim: Simulator,
    hfov,
    task_goal,
    goals_by_category,
    top_down_map,
    fog_of_war_mask,
    area_thres_in_pixels,
    visibility_dist_in_pixels,
    visited_frontiers,
    visible_ids,
    ins_list,
    vis_frontier = False,
    strategy = 'best',
    prev_prompt = "",
):
    task_goal_type = task_goal[1]
    task_goal_category = task_goal[0]
    task_goal_inst_id = task_goal[2]
    prompt = prev_prompt + " " +task_goal[3]

    target_position = None
    target_goal_id = None
    tgt_xy = np.array([0, 0])

    goal_positions = []
    goal_category = []
    decision_list = []

    goal_list = []
    # prompt = ""

    start_position = sim.get_agent_state().position

    assert task_goal_type == 'description'
    goal_list.append(goals_by_category[task_goal_category][task_goal_inst_id])


    for goal in goal_list:
        goal_position = goal['position']
        view_points = goal['view_points']
        min_dist = np.inf
        for view_point in view_points:
            view_point = view_point['agent_state']['position']
            cost = path_dist_cost(start_position, view_point, sim._simulator)
            if cost < min_dist:
                min_dist = path_dist_cost(start_position, view_point, sim._simulator)
                goal_position = view_point
        if min_dist == np.inf:
            continue
        goal["snap_position"] = goal_position
        goal_positions.append(goal_position)
        goal_category.append(goal)
    
    if len(goal_positions) == 0:
        raise ValueError("Goal not Navigable")

    goal_id_set = set()
    goal_id_dict = {}
    for goal in goal_category:
        goal_id = goal['object_id']
        goal_id = int(goal_id.split("_")[-1])
        goal_id_set.add(goal_id)
        goal_id_dict[goal_id] = goal

    goal_positions = np.array(goal_positions)

    def update_decision_list(frontier_list, best_idx, best_object=[], prompt = ""):
        cur_dict = {}
        cur_dict['time_step'] = len(ins_list) - 1
        cur_dict['object_list'] = list(visible_ids) 
        cur_dict['frontier_list'] = frontier_list.tolist()
        cur_dict['best_frontier_idx'] = int(best_idx)
        cur_dict['is_object_decision'] = len(best_object) > 0
        cur_dict['best_object_idx'] = best_object
        cur_dict['sentence'] = prompt
        cur_dict['object_list'] = [int(x) for x in cur_dict['object_list']]
        decision_list.append(cur_dict)

    def update_frontier():
        nonlocal fog_of_war_mask, top_down_map
        agent_state = sim.get_agent_state()
        fog_of_war_mask = reveal_fog_of_war(top_down_map=top_down_map, 
                                            current_fog_of_war_mask=fog_of_war_mask, 
                                            current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim._simulator), 
                                            current_angle=get_polar_angle(agent_state), 
                                            fov=hfov, max_line_len=visibility_dist_in_pixels, 
                                            enable_debug_visualization=vis_frontier)


    def find_closest_target():
        nonlocal target_position, target_goal_id, tgt_xy
        agent_state = sim.get_agent_state()
        if goal_positions.shape[0] > 1:  
            goal_idx, min_cost = astar_search(goal_positions, agent_state.position, sim._simulator)
            if goal_idx is None:
                raise ValueError("No valid goal")
        else:
            goal_idx = 0
        target_position = goal_category[goal_idx]['snap_position']
        target_goal_id = goal_category[goal_idx]['object_id']
        tgty, tgtx = map_coors_to_pixel(target_position, top_down_map, sim._simulator)
        tgt_xy = np.array([tgtx, tgty])

    def find_closest_visited_frontier():
        if len(visited_frontiers) == 0:
            return np.inf
        goal_idx, min_cost = astar_search(np.array(visited_frontiers), target_position, sim._simulator)
        return min_cost

    def choose_frontier_waypoint(gen_mode="best"):
        nonlocal tgt_xy
        assert gen_mode in ["closest", "best", "mixed"]
        update_frontier()
        agent_state = sim.get_agent_state()
        frontier_waypoints = detect_frontier_waypoints(top_down_map, 
                                fog_of_war_mask, area_thres_in_pixels,
                                tgt_xy=tgt_xy, 
                                xy=map_coors_to_pixel(agent_state.position, top_down_map, sim._simulator)[::-1], 
                                enable_visualization=vis_frontier)
        if len(frontier_waypoints) == 0:
            return np.array([]), -1, np.inf, -1
        frontier_waypoints = frontier_waypoints[:, ::-1] # convert to fun x,y space
        frontier_list = pixel_to_map_coors(frontier_waypoints, agent_state.position, top_down_map, sim._simulator)
        if len(visited_frontiers) > 0:
            new_list = []
            for idx, waypoint in enumerate(frontier_waypoints):
                if any(np.array_equal(frontier_list[idx], tmp) for tmp in visited_frontiers):
                    continue
                new_list.append(waypoint)
            if len(new_list) > 0:
                frontier_waypoints = np.array(new_list)
            else:
                return np.array([]), -1, np.inf, -1
        # search for closed
        closest_idx, _ = get_closest_waypoint(frontier_waypoints=frontier_waypoints, agent_position=agent_state.position, top_down_map=top_down_map, sim=sim._simulator)
        best_idx, best_dist = get_closest_waypoint(frontier_waypoints=frontier_waypoints, agent_position=target_position, top_down_map=top_down_map, sim=sim._simulator)

        frontier_list = pixel_to_map_coors(frontier_waypoints, agent_state.position, top_down_map, sim._simulator)

        if gen_mode == "closest":
            chosen_idx = closest_idx
        elif gen_mode == "best":
            chosen_idx = best_idx
        else:
            if np.random.rand() < 0.5:
                chosen_idx = best_idx
            else:
                chosen_idx = closest_idx

        return frontier_list, best_idx, best_dist, chosen_idx

    def observations_by_actions(action_list):
        dt = 1 / sim._fps
        for action in action_list:
            obs = sim.step(action)

            semantic_1st = obs[0]["semantic_1st"]

            unique_ids = set(np.unique(semantic_1st))
            visible_ids.update(unique_ids)

            ins_list.append(semantic_1st[...])
        
            update_frontier()

    def goto(goal):
        action_list = sim.goto_action(goal)
        if len(action_list) > 0 and action_list[-1] is None:
            action_list = action_list[:-1]
        observations_by_actions(action_list)

    def spin_around():
        action_list = ['turn_left'] * 12
        observations_by_actions(action_list)

    # target on the map
    def find_target(tpx, tpy):
        dx = [-1, 0, 1]
        dy = [-1, 0, 1]
        for i in range(3):
            for j in range(3):
                cur_x = tpx + dx[i]
                cur_y = tpy + dy[j]
                if not 0 <= cur_x < top_down_map.shape[0] or not 0 <= cur_y < top_down_map.shape[1]:
                    continue
                if fog_of_war_mask[cur_x, cur_y] > 0:
                    return True
        return False
    
    status = 'normal'
    while True:
        spin_around()
        #check if goal is visible
        have_seen = list(visible_ids & goal_id_set)
        if len(have_seen) > 0:
            #check reachable
            reachable_list = []
            reachable_positions = []
            for goal_id in have_seen:
                goal_position = goal_id_dict[goal_id]['snap_position']
                tgty, tgtx = map_coors_to_pixel(goal_position, top_down_map, sim._simulator)
                if find_target(tgty, tgtx):
                    reachable_list.append(goal_id)
                    reachable_positions.append(goal_position)
            if len(reachable_list) > 0:
                reachable_positions = np.array(reachable_positions)
                if len(reachable_list) == 1:
                    goal_idx =0
                else:
                    goal_idx, _ = astar_search(reachable_positions, sim.get_agent_state().position, sim._simulator)
                target_position = reachable_positions[goal_idx]
                tgty, tgtx = map_coors_to_pixel(target_position, top_down_map, sim._simulator)
                tgt_xy = np.array([tgtx, tgty])
                frontier_list, best_idx, best_dist, chosen_idx = choose_frontier_waypoint(strategy)
                update_decision_list(frontier_list, best_idx, [goal_id_dict[goal_id]['object_id'] for goal_id in reachable_list], prompt=prompt)
                goto(target_position)
                break

        find_closest_target()
        prev_best_dist = find_closest_visited_frontier()
        frontier_list, best_idx, best_dist, chosen_idx = choose_frontier_waypoint(strategy)
        better_possiblity = best_dist + 1e-6 < prev_best_dist
        visible = int(target_goal_id.split("_")[-1]) in visible_ids
        in_map = find_target(tgt_xy[1], tgt_xy[0])
        if visible and not in_map:
            if better_possiblity:
                update_decision_list(frontier_list, best_idx, [], prompt=prompt)
                next_frontier = frontier_list[chosen_idx]
                visited_frontiers.append(next_frontier)
                goto(next_frontier)
                continue
            else:
                update_decision_list(frontier_list, best_idx, [], prompt=prompt)
                goto(target_position)
                status = 'not_in_map'
                break
        elif not visible and in_map:
            if better_possiblity:
                update_decision_list(frontier_list, best_idx, [], prompt=prompt)
                next_frontier = frontier_list[chosen_idx]
                visited_frontiers.append(next_frontier)
                goto(next_frontier)
                continue
            else:
                status = 'not_visible'
                break
        elif not visible and not in_map:
            if better_possiblity:
                update_decision_list(frontier_list, best_idx, [], prompt=prompt)
                next_frontier = frontier_list[chosen_idx]
                visited_frontiers.append(next_frontier)
                goto(next_frontier)
                continue
            else:
                status = 'unable_to_explore'
                break
        else:
            raise ValueError("Unknown status")


    decision_json = {
        "strategy": strategy,
        "status": status,
        # "episode_info": cur_episode,
        "decision_list": decision_list,
    }

    return decision_json, fog_of_war_mask, prompt



def main(args):
    np.random.seed(100)
    input_scene_dir = args.input_scene_dir
    input_scene_dir = os.path.join(input_scene_dir, args.split)
    scene_list = os.listdir(input_scene_dir)
    valid_scene_list = []
    for scene_id in scene_list:
        if '.json' in scene_id:
            continue
        if not os.path.exists(f"data/sg3d/{args.split}/content/{scene_id.split('-')[1]}.json.gz"):
            continue
        if os.path.exists(os.path.join(input_scene_dir, scene_id, scene_id.split('-')[1] + '.semantic.txt')):
            valid_scene_list.append(scene_id)
    valid_scene_list.sort()
    # valid_scene_list = valid_scene_list[:5]
    # valid_scene_list.append('00407-NPHxDe6VeCc')
    print(f"Total scene number: ", len(valid_scene_list))
    start_time = datetime.datetime.now()
    print("Start Time of ALL", start_time)

    mp.set_start_method("forkserver", force=True) 
 
    with mp.Pool(args.num_workers) as pool:
        results = pool.starmap_async(handle_scene, [(x, args) for x in valid_scene_list], args.chunksize)
        # results.wait()
        tmp = results.get()

    end_time = datetime.datetime.now()
    print("End Time of ALL", end_time, "Total time", end_time - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_scene_dir", type=str, default='data/hm3d')
    parser.add_argument("--output_dir", type=str, default='tmp/sg3d')
    parser.add_argument("--split", type=str, default='val') # train, val
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--chunksize", type=int, default=1)
    args = parser.parse_args()
    main(args)

