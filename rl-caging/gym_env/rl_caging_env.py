import gymnasium as gym
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from read_state import ReadState
from typing import Optional
from gazebo_services import GazeboControl

import functools
import random
from copy import copy
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
import concurrent.futures


class RLCagingEnv(ParallelEnv):

    metadata = {
        "name": "rl_caging_env_v0",
    }

    def __init__(self):
        super(RLCagingEnv, self).__init__()

        self.agents = [f"diff_drive_robot{i+1}" for i in range(6)]
        self.possible_agents = self.agents[:]
        
        # Initialize ROS2 node
        rclpy.init()
        self.node = rclpy.create_node('rl_caging_env_node')

        # Initialize Gazebo control
        self.gazebo_control = GazeboControl()

        # Publishers for each robot
        self.cmd_vel_publishers = {
            agent: self.node.create_publisher(
                Twist,
                f"/{agent}/cmd_vel",
                10
            )
            for agent in self.agents
        }

        # Create read_state subscriber to get observations
        self.read_state_node = ReadState()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Initialize random generator
        if not hasattr(self, "np_random"):
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # --- Distance constraints ---
        min_dist = 0.5          # min distance from object/target
        min_agent_dist = 0.5    # min distance between agents

        # --- Sample object location ---
        self._object_location = self.np_random.uniform(low=-2, high=2, size=(2,)).astype(np.float32)

        # --- Sample target location (ensure far enough from object) ---
        while True:
            self._target_location = self.np_random.uniform(low=-2, high=2, size=(2,)).astype(np.float32)
            dist = np.linalg.norm(self._target_location - self._object_location)
            if dist >= min_dist:
                break

        # --- Sample agent locations ---
        self._agent_locations = {}

        for agent in self.agents:
            max_attempts = 1000
            for _ in range(max_attempts):
                loc = self.np_random.uniform(low=-2, high=2, size=(2,)).astype(np.float32)

                dist_obj = np.linalg.norm(loc - self._object_location)
                dist_target = np.linalg.norm(loc - self._target_location)

                too_close = any(
                    np.linalg.norm(loc - other_loc) < min_agent_dist
                    for other_loc in self._agent_locations.values()
                )

                if dist_obj >= min_dist and dist_target >= min_dist and not too_close:
                    self._agent_locations[agent] = loc
                    break
            else:
                self.node.get_logger().warn(f"Failed to find valid spawn for {agent}, placing randomly.")
                self._agent_locations[agent] = self.np_random.uniform(low=-2, high=2, size=(2,)).astype(np.float32)

        # --- Set all poses in parallel ---
        self._set_all_entity_poses_parallel()

        # --- Gather initial observations and infos ---
        observations = self._get_observations()
        infos = self._get_infos()

        return observations, infos
            
    def step(self, actions):

        for agent, action in actions.items():
            twist_msg = Twist()
            twist_msg.linear.x = float(action[0])
            twist_msg.angular.z = float(action[1])
            self.cmd_vel_publishers[agent].publish(twist_msg)

        # Unpause simulation
        self.gazebo_control.unpause_simulation()

        # let simulation run for a fixed simulation time step
        self.read_state_node.wait_for_new_scan()
        
        # Get observations and infos
        observations = self._get_observations()
        infos = self._get_infos()

        # Pause simulation
        self.gazebo_control.pause_simulation()

        # Placeholder rewards and done flags
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # ============================================
        # --- Collision / Too-Close Termination Check
        # ============================================
        min_separation = 0.200  # meters

        for agent, obs in observations.items():
            # obs structure:
            # [object_dist, object_angle, target_dist, target_angle,
            #  robot1_dist, robot1_angle, robot2_dist, robot2_angle]
            robot1_dist = obs[4]
            robot2_dist = obs[6]

            # If any detected robot is too close
            if (0.0 < robot1_dist <= min_separation) or (0.0 < robot2_dist <= min_separation):
                terminations[agent] = True
                rewards[agent] -= 1.0  # optional penalty for being too close

        return observations, rewards, terminations, truncations, infos
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Obervation space:
        # [object_dist, object_angle, agent1_dist, agent1_angle, agent2_dist, agent2_angle]
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([10.0, 360.0, 10.0, 360.0, 10.0, 360.0], dtype=np.float32)
        observation_space = Box(low=low, high=high, dtype=np.float32)
        return observation_space 

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Action space: 
        # [linear_velocity, angular_velocity]
        low = np.array([0.0, -2.0], dtype=np.float32)
        high = np.array([0.5, 2.0], dtype=np.float32)
        action_space = Box(low=low, high=high, dtype=np.float32)
        return action_space
    
    def _get_observations(self):
        return self.read_state_node.get_states()

    def _get_infos(self):
        return self.read_state_node.get_infos()
    
    def _set_all_entity_poses_parallel(self):
        """Set positions for all agents, object, and target in parallel to speed up reset."""
        entities = list(self._agent_locations.items()) + [
            ("object", self._object_location),
            ("target", self._target_location),
        ]

        def set_pose(entity_name, loc):
            return self.gazebo_control.set_entity_pose(
                entity_name=entity_name,
                x=float(loc[0]),
                y=float(loc[1])
            )

        # Run all set_pose calls concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(entities)) as executor:
            futures = [executor.submit(set_pose, name, loc) for name, loc in entities]
            concurrent.futures.wait(futures)