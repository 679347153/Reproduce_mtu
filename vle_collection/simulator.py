from typing import (TYPE_CHECKING, Any, Dict, Iterator, List, Optional,
                    Sequence, Tuple, Union, overload)
from omegaconf import DictConfig
import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim import Simulator as Sim
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum
from utils.functions import *
from utils.agent import Agent
from utils.make_config import make_agent_cfg, make_sim_cfg


class Simulator:
    """simulator related"""
    _resolution: List[int] # resolution of all sensors
    _fps: int # each step in the simulator equals to (1/fps) secs in the simulated world
    _config: DictConfig # used to initialize the habitat-sim simulator
    _simulator: Sim # habitat_sim simulator
    
    """agent related"""
    agents: List[Agent] # store agent states during simulation
    num_of_agents: int # number of agents in the simulator
    _agent_object_ids: List[int] # the object_ids of the rigid objects attached to the agents 
    _default_agent_id: int # 0 for the default agent


    def __init__(self, sim_settings, agents_settings) -> None:
        self._resolution = [sim_settings['height'], sim_settings['width']]
        self._fps = sim_settings['fps']
        self.agents = []
        agent_configs = []
        for i in range(len(agents_settings)):
            agent_configs.append(make_agent_cfg(self._resolution, agents_settings[i]))
            self.agents.append(Agent(self, agents_settings[i], i))
        self._config = make_sim_cfg(sim_settings, agent_configs)
        self._simulator = Sim(self._config)
        self.num_of_agents = len(self.agents)
        self._agent_object_ids = [None for _ in range(self.num_of_agents)] #by default agents have no rigid object bodies

        self._default_agent_id = sim_settings["default_agent"]

  
    
    def __del__(self):
        self._simulator.close()

    def get_observations(self):
        """get all observations"""
        obs = self._simulator.get_sensor_observations()
        return obs


    def get_agent_state(self, agent_id=0):
        """get the agent state given agent_id"""
        state = self._simulator.agents[agent_id].get_state()
        return state


    def set_agent_state(self, state, agent_id=0):
        """set the agent state given agent_id"""
        state = self._simulator.agents[agent_id].set_state(state)
        return state


    def get_path_finder(self):
        """get the habitat-sim pathfinder"""
        return self._simulator.pathfinder


    def get_island(self, position):
        """get the navmesh id of the position"""
        return self.get_path_finder().get_island(position)


    def unproject(self, agent_id=None):
        """get the crosshair ray of the agent"""
        if agent_id is None:
            agent_id = self._default_agent_id
        sensor = self._simulator.get_agent(agent_id)._sensors["semantic_1st"]
        view_point = mn.Vector2i(self._resolution[1]//2, self._resolution[0]//2)
        ray = sensor.render_camera.unproject(view_point, normalized=True)
        return ray


    def get_camera_info(self, camera_name, agent_id=None):
        if agent_id is None:
            agent_id = self._default_agent_id
        sensor = self._simulator.get_agent(agent_id)._sensors["semantic_1st"]
        camera = sensor.render_camera
        print('camera matrix: ', camera.camera_matrix)
        print('node: ', camera.node)
        print('projection matrix: ', camera.projection_matrix)
        return camera.camera_matrix, camera.projection_matrix

    def goto_action(self, target_position, agent_id=0):
        """return an action list to goto someplace"""
        action_list = []
        actions = self.reset_to_horizontal(agent_id=agent_id)
        action_list += actions[:-1]
        agent_state = self.get_agent_state(agent_id)
        agent_position = agent_state.position
        agent_island = self.get_island(agent_position)
        #project the target position to the agent's navmesh island
        path_finder = self.get_path_finder()
        target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
        follower = habitat_sim.GreedyGeodesicFollower(
            path_finder,
            self._simulator.agents[agent_id],
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right")
        try:
            action_list += follower.find_path(target_on_navmesh)
        except:
            pass
        return action_list

    def look_action(self, target_position, agent_id=0):
        """return an action list to look at someplace"""
        camera_ray = self.unproject(agent_id)
        #get the camera's center pixel position, ray direction, and target direction
        camera_position = np.array(camera_ray.origin)
        camera_direction = np.array(camera_ray.direction)
        target_direction = np.array(target_position)-camera_position
        target_direction = target_direction / np.linalg.norm(target_direction)
        action_list = []
        #initialize the inner product
        max_product = np.dot(target_direction, camera_direction)
        y_axis = [0.0, 1.0, 0.0]
        #greedy algorithm for maximizing the inner product of the camera ray and the target direction
        #first try to turn left and right
        while True:
            step = None
            current_camera_direction = None
            for action in ['turn_left', 'turn_right']:
                degree = self.agents[agent_id].action_space[action]
                if action == 'turn_left':
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=y_axis, radian=degree/180*np.pi)
                if action == 'turn_right':
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=y_axis, radian=-degree/180*np.pi)
                product = np.dot(new_camera_direction, target_direction)
                if product > max_product:
                    max_product = product
                    current_camera_direction = new_camera_direction
                    step = action
            if step == None:
                break
            camera_direction = current_camera_direction
            action_list.append(step)
        # then try look up and down    
        while True:
            step = None
            current_camera_direction = None
            for action in ['look_up', 'look_down']:
                degree = self.agents[agent_id].action_space[action]
                if action == 'look_up':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=-degree/180*np.pi)
                if action == 'look_down':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=degree/180*np.pi)
                product = np.dot(new_camera_direction, target_direction)
                if product > max_product:
                    max_product = product
                    current_camera_direction = new_camera_direction
                    step = action
            camera_direction = current_camera_direction
            action_list.append(step)
            if step == None:
                break
        return action_list
    

    def reset_to_horizontal(self, agent_id):
        """return an action list to look horizontally"""
        camera_ray = self.unproject(agent_id)
        #get the camera's direction
        camera_direction = np.array(camera_ray.direction)
        y_axis = [0.0, 1.0, 0.0]
        min_product = abs(np.dot(y_axis, camera_direction))
        #greedy algorithm for minimizing the abs(product)
        action_list = []
        while True:
            step = None
            current_camera_direction = None
            for action in ['look_up', 'look_down']:
                degree = self.agents[agent_id].action_space[action]
                if action == 'look_up':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=-degree/180*np.pi)
                if action == 'look_down':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=degree/180*np.pi)
                product = abs(np.dot(new_camera_direction, y_axis))
                if product < min_product:
                    min_product = product
                    current_camera_direction = new_camera_direction
                    step = action
            camera_direction = current_camera_direction
            action_list.append(step)
            if step == None:
                break
        return action_list

    
    def perform_discrete_collision_detection(self):
        """perform discrete collision detection for the scene"""
        self._simulator.perform_discrete_collision_detection()
    

    def get_physics_contact_points(self):
        """return a list of ContactPointData ” “objects describing the contacts from the most recent physics substep"""
        return self._simulator.get_physics_contact_points()



    def is_agent_colliding(self, agent_id, action):
        """ check wether the action will cause collision. Used to avoid border conditions during simulation. """
        if action not in ["move_forward", "move_backward"]: #only move action will cause collision
            return False
        step_size = self.agents[agent_id].step_size
        agent_transform = self._simulator.agents[agent_id].body.object.transformation
        if action == "move_forward":
            position = - agent_transform.backward * step_size
        else:
            position = agent_transform.backward * step_size

        new_position = agent_transform.translation + position
        filtered_position = self.get_path_finder().try_step(
            agent_transform.translation,
            new_position)
        dist_moved_before_filter = (new_position - agent_transform.translation).dot()
        dist_moved_after_filter = (filtered_position - agent_transform.translation).dot()
        # we check to see if the the amount moved after the application of the filter
        # is _less_ than the amount moved before the application of the filter
        EPS = 1e-4
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        return collided

    def step(self, actions: Union[str, dict, None]):
        """all agents perform actions in the environment and return observations."""
        if actions == None:
            actions = {self._default_agent_id: "no_op"}
        assert type(actions) in [str, dict]
        if type(actions) is str: #a single action for the default agent
            actions = {self._default_agent_id: actions}
        for agent_id in actions:
            action = actions[agent_id]
            assert action in self.agents[agent_id].action_space
            # agent_position = self.get_agent_state().position
        observations = self._simulator.step(action=actions, dt=1/self._fps)
        return observations

    def initialize_agent(self, agent_id=0, position=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0, 1.0]):
        """initialize an agent by its position and rotation"""
        agent_state = habitat_sim.AgentState()
        agent_state.position = position
        agent_state.rotation = rotation
        agent = self._simulator.initialize_agent(agent_id, agent_state)
        return agent.scene_node.transformation_matrix()

      
    def randomly_initialize_agent(self, agent_id=0):
        """randomly initialize an agent"""
        point = self.get_path_finder().get_random_navigable_point(max_tries=10, island_index=0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = point
        agent_state.rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)
        agent = self._simulator.initialize_agent(agent_id, agent_state)
        return agent.scene_node.transformation_matrix()


    def reconfigure(self, sim_settings, agents_settings):
        """reconfigure"""
        self._resolution = [sim_settings['height'], sim_settings['width']]
        self._fps = sim_settings['fps']
        self.agents = []

        agent_configs = []
        for i, single_agent_settings in enumerate(agents_settings):
            agent_configs.append(make_agent_cfg(self._resolution, single_agent_settings))
            self.agents.append(Agent(self, single_agent_settings, i))

        self.num_of_agents = len(self.agents)
        self._agent_object_ids = [None for _ in range(self.num_of_agents)] #by default agents have no rigid object bodies
        self._config = make_sim_cfg(sim_settings, agent_configs)
        self._simulator.reconfigure(self._config)

        self._default_agent_id = sim_settings["default_agent"]


    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
        episode=None) -> float:
        """shortest distance from a to b"""
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], (Sequence, np.ndarray)): #multiple endpoints
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else: #single endpoints
                path.requested_ends = np.array(
                    [np.array(position_b, dtype=np.float32)]
                )
        else:
            path = episode._shortest_path_cache
        path.requested_start = np.array(position_a, dtype=np.float32)
        self.get_path_finder().find_path(path) #Finds the shortest path between a start point and the closest of a set of end points (in geodesic distance) on the navigation mesh using MultiGoalShortestPath module. Path variable is filled if successful. Returns boolean success.
        if episode is not None:
            episode._shortest_path_cache = path
        return path.geodesic_distance