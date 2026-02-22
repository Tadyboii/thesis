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

    def set_entity_pose(self, entity_name: str, x: float, y: float):
        entity_pose_req = (
            f'name: "{entity_name}" '
            f'position: {{x: {x}, y: {y}, z: 0.01}} '
            f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'
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
