import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose # Changed from PoseStamped
# from tf_transformations import euler_from_quaternion
import math

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    """
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
import pygame
from particle_box2d import DiffDrivePushEnv
import numpy as np
import threading
import time

class GazeboToBox2DMirror(Node):
    def __init__(self, n_agents=6):
        super().__init__('gazebo_box2d_mirror')
        self.n_agents = n_agents

        # --- Box2D Environment ---
        # Initialize with render_mode="human" to visualize
        self.env = DiffDrivePushEnv(n_agents=self.n_agents, render_mode="human")
        self.env.reset() # Setup bodies

        # --- State Storage ---
        self.robot_states = {}  # {agent_id: (x, y, theta)}
        self.object_state = None # (x, y, theta)
        self.target_state = None # (x, y)

        # --- Subscribers ---
        # Agents
        for i in range(self.n_agents):
            agent_id = f"diff_drive_robot{i+1}" # e.g. diff_drive_robot1
            # Adjust topic name if your bridge uses something else, e.g., /model/.../odometry
            topic = f"/{agent_id}/odom" 
            self.create_subscription(
                Odometry,
                topic,
                lambda msg, aid=i: self.robot_odom_callback(msg, aid),
                10
            )

        # Object (Assuming bridged topic /object/pose or similar)
        # If not available, you might need to bridge /model/object/pose
        self.create_subscription(
            Pose, # Changed from PoseStamped
            '/model/object/pose', 
            self.object_pose_callback,
            10
        )

        # Target (Assuming bridged topic /target/pose)
        self.create_subscription(
            Pose, # Changed from PoseStamped
            '/model/target/pose',
            self.target_pose_callback,
            10
        )
        
        self.get_logger().info("Mirror Node Started. Waiting for ROS messages...")

    def robot_odom_callback(self, msg: Odometry, agent_idx):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        
        # Store state
        self.robot_states[agent_idx] = (pos.x, pos.y, yaw)

    def object_pose_callback(self, msg: Pose): # Changed from PoseStamped
        pos = msg.position # No .pose nesting for Pose msg
        ori = msg.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.object_state = (pos.x, pos.y, yaw)

    def target_pose_callback(self, msg: Pose): # Changed from PoseStamped
        pos = msg.position
        self.target_state = (pos.x, pos.y)

    def update_box2d_world(self):
        # --- Update Agents ---
        for i, body in enumerate(self.env.agent_bodies):
            if i in self.robot_states:
                x, y, yaw = self.robot_states[i]
                # Update Box2D body position and angle directly
                body.position = (x, y)
                body.angle = yaw
                # Wake up body to ensure rendering/physics response if needed
                body.Awake = True 

        # --- Update Object ---
        if self.object_state and self.env.object_body:
            x, y, yaw = self.object_state
            self.env.object_body.position = (x, y)
            self.env.object_body.angle = yaw
            self.env.object_body.Awake = True

        # --- Update Target ---
        # Target in particle_box2d might just be a position rendering, check if it has a body
        # Looking at particle_box2d code (inferred), target is often just drawn or static sensor
        if self.target_state:
            # If target is stored as self.env.target_pos
            self.env.target_pos = np.array(self.target_state, dtype=np.float32)

    def render_loop(self):
        running = True
        while running and rclpy.ok():
            # Handle PyGame events
            if self.env.screen:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            
            # Sync Box2D with latest ROS data
            self.update_box2d_world()
            
            # Render
            self.env.render()
            
            # Rate limit rendering
            time.sleep(0.02) # ~50 FPS

        pygame.quit()
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    mirror_node = GazeboToBox2DMirror(n_agents=6)

    # Run ROS spin in a separate thread so render loop isn't blocked
    spin_thread = threading.Thread(target=rclpy.spin, args=(mirror_node,), daemon=True)
    spin_thread.start()

    try:
        mirror_node.render_loop()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
