#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_share = get_package_share_directory('rl_caging_simulation')

    # Declare headless argument (default: false → GUI visible)
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode (no GUI). Pass headless:=true to enable.'
    )
    headless = LaunchConfiguration('headless')

    # Launch the empty world, forwarding the headless flag
    empty_world_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'empty_world.launch.py')
        ),
        launch_arguments={'headless': headless}.items()
    )

    # Launch the spawn script
    spawn_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'spawn.launch.py')
        )
    )

    delayed_spawn_cmd = TimerAction(
        period=5.0,
        actions=[spawn_cmd]
    )

    ld = LaunchDescription()
    ld.add_action(headless_arg)
    ld.add_action(empty_world_cmd)
    ld.add_action(delayed_spawn_cmd)

    return ld
