import re
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
import subprocess

class GazeboControl(Node):
    def __init__(self):
        super().__init__('gazebo_control')

        self.sim_time = None
        self.subscription = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            10
        )

    def clock_callback(self, msg: Clock):
        # Convert ROS time message to seconds
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def get_sim_time(self):
        """Return latest simulation time, or None if not yet received."""
        if self.sim_time is None:
            self.get_logger().warn("Simulation time not received yet from /clock")
            return None
        return self.sim_time

    # def get_entity_pos(self, entity_name: str):
    #     """Get the (x, y) world position of a named entity from /world/car_world/pose/info.
    #     Returns (x, y) as floats, or (None, None) on failure."""
    #     try:
    #         result = subprocess.run(
    #             ["gz", "topic", "-e", "-t", "/world/car_world/pose/info", "--count", "1"],
    #             capture_output=True, text=True, timeout=3.0
    #         )
    #         # Split raw output into individual pose blocks
    #         blocks = result.stdout.split("pose {")
    #         for block in blocks:
    #             if f'name: "{entity_name}"' in block:
    #                 x_m = re.search(r'x:\s*([+-]?[\d.eE+-]+)', block)
    #                 y_m = re.search(r'y:\s*([+-]?[\d.eE+-]+)', block)
    #                 x = float(x_m.group(1)) if x_m else 0.0
    #                 y = float(y_m.group(1)) if y_m else 0.0
    #                 return (x, y)
    #     except Exception:
    #         pass
    #     return (None, None)

    def set_entity_pose(self, entity_name: str, x: float, y: float, z: float = 0.01, orientation_x: float = 0.0, orientation_y: float = 0.0, orientation_z: float = 0.0, orientation_w: float = 1.0):
        entity_pose_req = (
            f'name: "{entity_name}" '
            f'position: {{x: {x}, y: {y}, z: {z}}} '
            f'orientation: {{x: {orientation_x}, y: {orientation_y}, z: {orientation_z}, w: {orientation_w}}}'
        )
        result = subprocess.run([
            "gz", "service",
            "-s", "/world/car_world/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--req", entity_pose_req
        ], capture_output=True, text=True)
        return result.stdout, result.stderr

    def pause_simulation(self):
        result = subprocess.run([
            "gz", "service",
            "-s", "/world/car_world/playback/control",
            "--reqtype", "gz.msgs.LogPlaybackControl",
            "--reptype", "gz.msgs.Boolean",
            "--req", "pause: true"
        ], capture_output=True, text=True)
        return result.stdout, result.stderr

    def unpause_simulation(self):
        result = subprocess.run([
            "gz", "service",
            "-s", "/world/car_world/playback/control",
            "--reqtype", "gz.msgs.LogPlaybackControl",
            "--reptype", "gz.msgs.Boolean",
            "--req", "forward: true"
        ], capture_output=True, text=True)
        return result.stdout, result.stderr
