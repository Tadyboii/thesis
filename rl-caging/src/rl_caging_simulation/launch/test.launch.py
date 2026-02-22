from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Locate the packages that contain your sub-launch files
    pkg_tb3_sim = FindPackageShare('rl_caging_simulation').find('rl_caging_simulation')

    # Paths to the launch files you want to include
    launch_file_1 = os.path.join(pkg_tb3_sim, 'launch', 'empty_world.launch.py')
    launch_file_2 = os.path.join(pkg_tb3_sim, 'launch', 'multi_turtlebot3.launch.py')

    # Include both
    include_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file_1)
    )

    include_2 = TimerAction(
        period=5.0, 
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(launch_file_2)
            )
        ]
    )

    # Return the LaunchDescription
    return LaunchDescription([
        include_1,
        include_2
    ])