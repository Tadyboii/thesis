from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import math
import numpy as np


class ReadState(Node):

    def __init__(self, n_agents=6):
        super().__init__('read_state')
        self.get_logger().info("Read State Node has been started.")

        # --- Multi-agent setup ---
        self.num_agents = n_agents
        self.agents = [f"diff_drive_robot{i+1}" for i in range(self.num_agents)]
        self.agent_data = {
            agent: {
                "ranges": [],
                "intensities": [],
                "last_scan_time": None,
                "new_scan_received": False,

                # detection results per agent
                "object_distance": 0.0,
                "object_angle": 0.0,
                "target_distance": 0.0,
                "target_angle": 0.0,
                "robot1_distance": 0.0,
                "robot1_angle": 0.0,
                "robot2_distance": 0.0,
                "robot2_angle": 0.0,
            }
            for agent in self.agents
        }

        # --- LaserScan parameters ---
        self.angle_min = 0.00  # radians
        self.angle_increment = 0.0174533  # 1 degree in radians

        # --- Intensity thresholds ---
        self.object_intensity_threshold_max = 500.0
        self.object_intensity_threshold_min = 100.0
        self.target_intensity_value = 500.0
        self.target_intensity_tolerance = 10.0

        # --- Create subscriptions for each agent ---
        for i, agent in enumerate(self.agents, start=1):
            topic = f"/diff_drive_robot{i}/scan"
            self.create_subscription(
                LaserScan,
                topic,
                lambda msg, a=agent: self.scan_callback(msg, a),
                qos_profile=qos_profile_sensor_data
            )

    # ===============================
    # --- LiDAR Callback Functions ---
    # ===============================
    def scan_callback(self, msg, agent):
        data = self.agent_data[agent]
        data["ranges"] = msg.ranges
        data["intensities"] = msg.intensities
        data["last_scan_time"] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        data["new_scan_received"] = True

        # Process detection for that agent
        self.scan_obstacles(agent)

    def scan_obstacles(self, agent):
        data = self.agent_data[agent]
        scan_ranges = data["ranges"]
        scan_intensities = data["intensities"]

        robot_obstacles = []
        current_robot_obstacle_distance = []
        current_robot_obstacle_angle = []
        first_robot_obstacle_distance = []
        first_robot_obstacle_angle = []
        current_object_obstacle_distance = []
        current_object_obstacle_angle = []
        current_target_obstacle_distance = []
        current_target_obstacle_angle = []

        for i, distance in enumerate(scan_ranges):
            if i >= len(scan_intensities):
                continue

            intensity = scan_intensities[i]
            angle = math.degrees(self.angle_min + i * self.angle_increment)

            # --- Cylindrical object ---
            if self.object_intensity_threshold_min < intensity < self.object_intensity_threshold_max:
                current_object_obstacle_distance.append(distance)
                current_object_obstacle_angle.append(angle)

            # --- Target object ---
            elif abs(intensity - self.target_intensity_value) <= self.target_intensity_tolerance:
                current_target_obstacle_distance.append(distance)
                current_target_obstacle_angle.append(angle)

            # --- Robot obstacles ---
            elif not math.isinf(distance) and not math.isnan(distance) and intensity <= self.object_intensity_threshold_min:
                current_robot_obstacle_distance.append(distance)
                current_robot_obstacle_angle.append(angle)
            else:
                # End of cluster
                if current_robot_obstacle_distance:
                    avg_distance = sum(current_robot_obstacle_distance) / len(current_robot_obstacle_distance)
                    avg_angle = sum(current_robot_obstacle_angle) / len(current_robot_obstacle_angle)
                    robot_obstacles.append({
                        'angle': round(avg_angle, 2),
                        'distance': round(avg_distance, 3),
                        'num_points': len(current_robot_obstacle_distance),
                    })

                    if not first_robot_obstacle_distance:
                        first_robot_obstacle_distance = current_robot_obstacle_distance
                        first_robot_obstacle_angle = current_robot_obstacle_angle

                    current_robot_obstacle_distance = []
                    current_krobot_obstacle_angle = []

        # --- Handle last cluster ---
        if current_robot_obstacle_distance:
            if first_robot_obstacle_distance:
                current_robot_obstacle_distance = first_robot_obstacle_distance + current_robot_obstacle_distance
                current_robot_obstacle_angle = first_robot_obstacle_angle + current_robot_obstacle_angle
                avg_distance = sum(current_robot_obstacle_distance) / len(current_robot_obstacle_distance)
                avg_angle = sum(current_robot_obstacle_angle) / len(current_robot_obstacle_angle)
                robot_obstacles[0] = {
                    'angle': round(avg_angle, 2),
                    'distance': round(avg_distance, 3),
                    'num_points': len(current_robot_obstacle_distance),
                }
            else:
                avg_distance = sum(current_robot_obstacle_distance) / len(current_robot_obstacle_distance)
                avg_angle = sum(current_robot_obstacle_angle) / len(current_robot_obstacle_angle)
                robot_obstacles.append({
                    'angle': round(avg_angle, 2),
                    'distance': round(avg_distance, 3),
                    'num_points': len(current_robot_obstacle_distance),
                })

        # --- Helper for mean cluster ---
        def avg_cluster(distances, angles):
            if not distances:
                return 0.0, 0.0
            avg_distance = sum(distances) / len(distances)
            angles_rad = [math.radians(a) for a in angles]
            sin_sum = sum(math.sin(a) for a in angles_rad)
            cos_sum = sum(math.cos(a) for a in angles_rad)
            mean_angle_rad = math.atan2(sin_sum / len(angles_rad), cos_sum / len(angles_rad))
            mean_angle_deg = math.degrees(mean_angle_rad) % 360
            return round(avg_distance, 3), round(mean_angle_deg, 2)

        # --- Compute and store results (per agent only) ---
        object_distance, object_angle = avg_cluster(current_object_obstacle_distance, current_object_obstacle_angle)
        target_distance, target_angle = avg_cluster(current_target_obstacle_distance, current_target_obstacle_angle)

        robot_obstacles.sort(key=lambda obs: obs['distance'])
        nearest_two = robot_obstacles[:2]
        robot1_distance = nearest_two[0]['distance'] if len(nearest_two) > 0 else 0.0
        robot1_angle = nearest_two[0]['angle'] if len(nearest_two) > 0 else 0.0
        robot2_distance = nearest_two[1]['distance'] if len(nearest_two) > 1 else 0.0
        robot2_angle = nearest_two[1]['angle'] if len(nearest_two) > 1 else 0.0

        # --- Save all results to this agent only ---
        data["object_distance"] = object_distance
        data["object_angle"] = object_angle
        data["target_distance"] = target_distance
        data["target_angle"] = target_angle
        data["robot1_distance"] = robot1_distance
        data["robot1_angle"] = robot1_angle
        data["robot2_distance"] = robot2_distance
        data["robot2_angle"] = robot2_angle

    # =========================
    # --- Public API Methods ---
    # =========================
    def get_states(self):
        obs_dict = {}
        for agent in self.agents:
            data = self.agent_data[agent]
            obs = np.array([
                data.get("object_distance", 0.0),
                data.get("object_angle", 0.0),
                data.get("target_distance", 0.0),
                data.get("target_angle", 0.0),
                data.get("robot1_distance", 0.0),
                data.get("robot1_angle", 0.0),
                data.get("robot2_distance", 0.0),
                data.get("robot2_angle", 0.0)
            ], dtype=np.float32)

            # Replace NaNs with zeros
            obs = np.nan_to_num(obs, nan=0.0)
            
            obs_dict[agent] = obs
        return obs_dict

    def get_infos(self):
        info_dict = {}
        for agent in self.agents:
            data = self.agent_data[agent]
            info_dict[agent] = {
                "object": {"distance": data.get("object_distance", 0.0), "angle": data.get("object_angle", 0.0)},
                "target": {"distance": data.get("target_distance", 0.0), "angle": data.get("target_angle", 0.0)},
                "agent1": {"distance": data.get("robot1_distance", 0.0), "angle": data.get("robot1_angle", 0.0)},
                "agent2": {"distance": data.get("robot2_distance", 0.0), "angle": data.get("robot2_angle", 0.0)}
            }
        return info_dict

    # ===================================
    # --- Wait Until All Scans Arrived ---
    # ===================================
    def wait_for_new_scan(self):
        """Wait indefinitely until all agents receive at least one new LiDAR scan."""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(self.agent_data[a]["new_scan_received"] for a in self.agents):
                for a in self.agents:
                    self.agent_data[a]["new_scan_received"] = False
                return
