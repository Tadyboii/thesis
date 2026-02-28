import time

import gymnasium as gym
import numpy as np
import pygame
import rclpy
from geometry_msgs.msg import Twist
from read_state import ReadState
from gazebo_services import GazeboControl
from typing import Optional

import functools
import math
import concurrent.futures
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box

class DiffDrivePushEnv(ParallelEnv):

    metadata = {
        "name": "rl_caging_env_v0",
        "render_modes": ["human"],
    }
    AGENT_COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]

    def __init__(self, n_agents=6, max_steps=150, render_mode=None, **kwargs):
        super(DiffDrivePushEnv, self).__init__()

        self.n_agents = n_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_count = 0
        
        self.agents = [f"diff_drive_robot{i+1}" for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        
        # Initialize ROS2 node
        if not rclpy.ok():
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
        self.read_state_node = ReadState(n_agents=self.n_agents)

        # Parameters matching particle_box2d
        self.max_speed = 1.0
        self.turn_sensitivity = 3.0
        self.max_obs_dist = 10.0
        self.object_radius = 1.5  # Used for logical checks if needed

        # --- Reward Tracking ---
        self.last_rewards = [0.0] * n_agents
        self.agent_cumulative_rewards = [0.0] * n_agents

        # --- Pause State ---
        self.paused = False

        # --- Pygame Setup ---
        self.screen = None
        self.font = None
        if self.render_mode == "human":
            pygame.init()
            self.start_render()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Initialize random generator
        if not hasattr(self, "np_random"):
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.step_count = 0
        self.last_rewards = [0.0] * self.n_agents
        self.agent_cumulative_rewards = [0.0] * self.n_agents

        # --- Distance constraints ---
        # Actual collision radius of the diff-drive robot chassis as defined
        # in diff_drive_robot.sdf (chassis cylinder radius = 0.14 m).
        # The lidar drum (0.1016 m) and wheels (0.033 m radius) are all
        # smaller, so 0.14 m is the outermost collision extent.
        AGENT_RADIUS = 0.14

        # Minimum center-to-center clearance between an agent and the object.
        # Must be > (object_radius + agent_radius) so the agent body never
        # overlaps the object cylinder at spawn time.
        min_dist_from_object = self.object_radius + AGENT_RADIUS + 0.1  # 1.5 + 0.14 + 0.1 = 1.74 m

        # Minimum center-to-center clearance between an agent and the target,
        # and between two agents (small buffer is enough here).
        min_dist = 0.5          # min distance from target / generic fallback
        min_agent_dist = 0.5    # min distance between agents
        
        # Map dimensions for spawning
        map_w, map_h = 10.0, 10.0 # Assuming similar to Box2D
        safe_margin = 1.0

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

                # dist_obj uses min_dist_from_object to prevent spawning
                # inside or clipping the cylindrical object
                if dist_obj >= min_dist_from_object and dist_target >= min_dist and not too_close:
                    self._agent_locations[agent] = loc
                    break
            else:
                self.node.get_logger().warn(f"Failed to find valid spawn for {agent}, placing randomly.")
                self._agent_locations[agent] = self.np_random.uniform(low=-2, high=2, size=(2,)).astype(np.float32)

        # --- Set all poses in parallel ---
        self._set_all_entity_poses_parallel()

        # --- Clear buffers ---
        # Wait for fresh scan after reset
        # We unpause briefly to let the physics settle and get a scan
        # self.gazebo_control.unpause_simulation()
        self.read_state_node.wait_for_new_scan()
        
        # Replacement for continuous run:
        # import time
        # time.sleep(0.1) # Wait for poses to settle in Gazebo

        observations = self._get_observations()
        infos = self._get_infos()
        # self.gazebo_control.pause_simulation()

        if self.render_mode == "human":
            self.render(observations)

        return observations, infos
            
    def step(self, actions):
        self.step_count += 1
        
        # Apply actions
        for i, agent in enumerate(self.agents):
            if agent in actions:
                action = actions[agent]
            elif i < len(actions): # Handle list/array input if needed
                 action = actions[i]
            else:
                 continue

            # Action Mapping to Box2D env:
            # action[0]: forward/backward [-1, 1]
            # action[1]: rotation [-1, 1]
            
            forward = float(action[0])
            angular = float(action[1])
            
            twist_msg = Twist()
            # particle_box2d: vx = cos(theta) * forward * max_speed.
            # DiffDrive robot takes linear.x and angular.z.
            twist_msg.linear.x = forward * self.max_speed
            twist_msg.angular.z = angular * self.turn_sensitivity
            
            if agent in self.cmd_vel_publishers:
                self.cmd_vel_publishers[agent].publish(twist_msg)

        # Unpause simulation
        # self.gazebo_control.unpause_simulation()

        # let simulation run for a fixed simulation time step (wait for scan)
        self.read_state_node.wait_for_new_scan()

        # Replacement for continuous run:
        # Instead of waiting for a scan, just sleep for a tiny bit to control rate
        # import time
        # time.sleep(0.1) # Slow down so we can see movement (approx 20 Hz) 
        
        # Get observations and infos
        observations = self._get_observations()
        infos = self._get_infos()

        # Pause simulation
        # self.gazebo_control.pause_simulation()

        # --- Reward Computation ---
        # Fetch raw sensor state again (same scan already acquired above via wait_for_new_scan)
        raw_obs_dict = self.read_state_node.get_states()

        # Normalisation constants matching particle_box2d
        max_dist  = self.max_obs_dist   # 10.0 m  – same as particle_box2d
        max_dist2 = max_dist ** 2       # used to normalise the squared penalty

        rewards = {}
        for i, agent in enumerate(self.agents):
            raw = raw_obs_dict.get(agent, np.zeros(8))

            # ------------------------------------------------------------------
            # DISTANCE PENALTY  (mirrors particle_box2d reward logic)
            #
            # raw[0] : distance reported by the lidar/sensor from this robot to
            #          the object.  Unlike Box2D (where dist is center-to-center
            #          and both radii must be subtracted), the ROS sensor already
            #          measures the gap to the nearest surface, so raw[0] is
            #          used directly as dist_surface.
            #
            # Penalty is quadratic in surface distance, scaled to [-10, 0]:
            #   penalty = −(dist_surface² / max_dist²) × 10.0
            # → 0.0   when the robot is touching the object (dist_surface ≈ 0)
            # → −10.0 when the robot is max_dist (10 m) away
            # ------------------------------------------------------------------
            dist_surface = max(0.0, float(raw[0]))
            dist_penalty = -(dist_surface * dist_surface / max_dist2) * 10.0

            rewards[agent] = dist_penalty

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.step_count >= self.max_steps for agent in self.agents}

        # Store per-agent rewards for render
        for i, agent in enumerate(self.agents):
            r = rewards.get(agent, 0.0)
            self.last_rewards[i] = r
            self.agent_cumulative_rewards[i] += r

        if self.render_mode == "human":
            self.render(observations)

        return observations, rewards, terminations, truncations, infos
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Match particle_box2d: Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        return Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Match particle_box2d: Box(-1.0, 1.0, (2,), np.float32)
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    
    def _get_observations(self):
        # Get raw dict from read_state
        raw_obs_dict = self.read_state_node.get_states()
        processed_obs = {}
        
        # Helper to normalize angle to [-1, 1] (where 1 = pi)
        def norm_angle(deg):
            # Convert 0..360 to -180..180
            a = deg % 360.0
            if a > 180.0:
                a -= 360.0
            # Convert to radians and divide by pi -> [-1, 1]
            rad = math.radians(a)
            return rad / math.pi

        # Helper to normalize distance to [0, 1]
        def norm_dist(d):
            return min(d, self.max_obs_dist) / self.max_obs_dist

        for agent in self.agents:
            raw = raw_obs_dict.get(agent, np.zeros(8))
            
            # read_state returns: 
            # 0: obj_dist, 1: obj_angle
            # 2: tgt_dist, 3: tgt_angle
            # 4: r1_dist,  5: r1_angle
            # 6: r2_dist,  7: r2_angle
            
            # particle_box2d expects:
            # [diff_angle_1, diff_dist_1, diff_angle_2, diff_dist_2, 
            #  target_angle, target_dist, object_angle, object_surf_dist]
            
            # 1. Neighbor 1
            n1_dist = norm_dist(raw[4])
            n1_ang  = norm_angle(raw[5])
            
            # 2. Neighbor 2
            n2_dist = norm_dist(raw[6])
            n2_ang  = norm_angle(raw[7])
            
            # 3. Target
            t_dist = norm_dist(raw[2])
            t_ang  = norm_angle(raw[3])
            
            # 4. Object
            o_dist = norm_dist(raw[0])
            o_ang  = norm_angle(raw[1])
            
            obs_vec = np.array([
                n1_ang, n1_dist,
                n2_ang, n2_dist,
                t_ang, t_dist,
                o_ang, o_dist
            ], dtype=np.float32)
            
            processed_obs[agent] = obs_vec
            
        return processed_obs

    def _get_infos(self):
        return self.read_state_node.get_infos()
    
    def _set_all_entity_poses_parallel(self):
        """Set positions for all agents, object, and target in parallel to speed up reset."""
        entities = list(self._agent_locations.items()) + [
            ("object", self._object_location),
            ("target", self._target_location),
        ]

        def set_pose(entity_name, loc):
            # Randomize yaw
            yaw = self.np_random.uniform(-math.pi, math.pi)
            spawn_z = 0.2 if entity_name == "object" else 0.0

            return self.gazebo_control.set_entity_pose(
                entity_name=entity_name,
                x=float(loc[0]),
                y=float(loc[1]),
                z=spawn_z,  # Lift slightly to avoid floor collision clipping
                orientation_x=0.0,
                orientation_y=0.0,
                orientation_z=math.sin(yaw/2),
                orientation_w=math.cos(yaw/2)
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(entities)) as executor:
            futures = [executor.submit(set_pose, name, loc) for name, loc in entities]


    def start_render(self):
        screen_width = 1100
        screen_height = 600
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("RL Caging Environment Observations")
        self.font = pygame.font.Font(None, 24)

    def render(self, observations=None):
        if self.render_mode != "human":
            return

        if self.screen is None:
            self.start_render()

        # Handle events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.paused = not self.paused
                if self.paused:
                    self.gazebo_control.pause_simulation()
                else:
                    self.gazebo_control.unpause_simulation()

        # Blocking pause loop: keep window responsive while simulation is paused
        while self.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.paused = False
                    self.gazebo_control.unpause_simulation()
            pygame.time.wait(100)

        self.screen.fill((0, 0, 0))  # Black background

        # Draw Header
        header_text = self.font.render(f"Step: {self.step_count} (Max: {self.max_steps})", True, (255, 255, 255))
        self.screen.blit(header_text, (10, 10))

        # Definitions
        # Obs: [n1_ang, n1_dist, n2_ang, n2_dist, t_ang, t_dist, o_ang, o_dist]
        column_headers = ["Agent", "N1_Ang", "N1_Dist", "N2_Ang", "N2_Dist", "T_Ang", "T_Dist", "O_Ang", "O_Dist", "Reward", "", "Total"]
        
        start_x = 20
        start_y = 50
        line_height = 30
        col_width = 80
        # X position where the reward bar starts (right after the last obs column)
        reward_col_x = start_x + 9 * col_width  # = 740

        # Draw Table Headers
        for i, header in enumerate(column_headers):
            text_surf = self.font.render(header, True, (255, 255, 255))
            self.screen.blit(text_surf, (start_x + i * col_width, start_y))

        if observations is None:
             pygame.display.flip()
             return

        # Draw Agent Data
        for idx, agent in enumerate(self.agents):
            y_pos = start_y + (idx + 1) * line_height
            agent_color = self.AGENT_COLORS[idx % len(self.AGENT_COLORS)]

            # Agent Name
            agent_name_surf = self.font.render(f"agent_{idx+1}", True, agent_color)
            self.screen.blit(agent_name_surf, (start_x, y_pos))

            # Observations values — same color as agent
            agent_obs = observations.get(agent, [])

            for obs_idx, val in enumerate(agent_obs):
                val_str = f"{val:.2f}"
                text_surf = self.font.render(val_str, True, agent_color)
                self.screen.blit(text_surf, (start_x + (obs_idx + 1) * col_width, y_pos))

            # --- Reward Bar inline (continues as extra columns after observations) ---
            r_val = self.last_rewards[idx] if idx < len(self.last_rewards) else 0.0
            cum_val = self.agent_cumulative_rewards[idx] if idx < len(self.agent_cumulative_rewards) else 0.0

            bar_x = reward_col_x
            bar_y = y_pos + 3          # vertically centred in the row
            bar_width = 130
            bar_height = 12
            zero_x = bar_x + bar_width - 30   # zero-line offset rightward for negative room
            scale = 80.0               # pixels per reward unit

            # Background
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            # Zero marker
            pygame.draw.line(self.screen, (150, 150, 150),
                             (zero_x, bar_y), (zero_x, bar_y + bar_height), 1)

            bar_len = int(abs(r_val) * scale)
            if r_val < 0:
                rect_x = max(bar_x, zero_x - bar_len)
                actual_len = zero_x - rect_x
                pygame.draw.rect(self.screen, (255, 50, 50),
                                 (rect_x, bar_y, actual_len, bar_height))
            else:
                capped_len = min(bar_len, bar_x + bar_width - zero_x)
                pygame.draw.rect(self.screen, (50, 255, 50),
                                 (zero_x, bar_y, capped_len, bar_height))

            # Numeric reward value (right of bar)
            r_lbl = self.font.render(f"{r_val:.3f}", True, agent_color)
            self.screen.blit(r_lbl, (bar_x + bar_width + 5, y_pos))

            # Cumulative total (further right)
            c_lbl = self.font.render(f"Tot:{cum_val:.1f}", True, agent_color)
            self.screen.blit(c_lbl, (bar_x + bar_width + 75, y_pos))

        # # --- Object-to-Target distance from Gazebo pose info ---
        # obj_pos = self.gazebo_control.get_entity_pos("object")
        # tgt_pos = self.gazebo_control.get_entity_pos("target")

        # if obj_pos[0] is not None and tgt_pos[0] is not None:
        #     dist_m = math.hypot(tgt_pos[0] - obj_pos[0], tgt_pos[1] - obj_pos[1])
        #     dist_str  = f"Obj-Target Dist: {dist_m:.3f} m"
        #     obj_str   = f"Object:  ({obj_pos[0]:.2f}, {obj_pos[1]:.2f})"
        #     tgt_str   = f"Target:  ({tgt_pos[0]:.2f}, {tgt_pos[1]:.2f})"
        # else:
        #     dist_str = "Obj-Target Dist: N/A"
        #     obj_str  = "Object:  N/A"
        #     tgt_str  = "Target:  N/A"

        # sw, sh = self.screen.get_size()
        # dist_surf = self.font.render(dist_str, True, (255, 255, 255))
        # obj_surf  = self.font.render(obj_str,  True, (180, 180, 255))
        # tgt_surf  = self.font.render(tgt_str,  True, (180, 255, 180))

        # line_h = dist_surf.get_height() + 4
        # base_y = sh - line_h * 3 - 8
        # for surf, dy in [(obj_surf, 0), (tgt_surf, line_h), (dist_surf, line_h * 2)]:
        #     self.screen.blit(surf, (sw - surf.get_width() - 10, base_y + dy))

        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
        if hasattr(self, 'node') and self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
