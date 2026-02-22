#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Joep Tool, Hyungyu Kim

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, SetEnvironmentVariable
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = os.path.join(
        get_package_share_directory('rl_caging_simulation'),
        'worlds',
        'world.world'
    )

    # Accept headless argument forwarded from simulation.launch.py
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode (no GUI).'
    )
    headless = LaunchConfiguration('headless')

    # Server – normal mode (with GUI client)
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world], 'on_exit_shutdown': 'true'}.items(),
        condition=UnlessCondition(headless)
    )

    # Server – headless mode (no display required)
    gzserver_headless_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s --headless-rendering -v2 ', world], 'on_exit_shutdown': 'true'}.items(),
        condition=IfCondition(headless)
    )

    # GUI client – only launched when NOT in headless mode
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 '}.items(),
        condition=UnlessCondition(headless)
    )

    set_env_vars_resources = AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(
                get_package_share_directory('rl_caging_simulation'),
                'models'))

    # Disable Gazebo Fuel internet lookups so the sim works fully offline.
    # GZ_FUEL_SERVER_CONFIG_PATH is pointed at a local yaml with servers: []
    # which prevents gz-fuel-tools from contacting fuel.gazebosim.org.
    disable_fuel_lookup = SetEnvironmentVariable(
        name='GZ_FUEL_SERVER_CONFIG_PATH',
        value=os.path.join(
            get_package_share_directory('rl_caging_simulation'),
            'config',
            'fuel_tools.yaml'
        )
    )
    
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
    )

    ld = LaunchDescription()

    ld.add_action(headless_arg)
    ld.add_action(disable_fuel_lookup)
    ld.add_action(set_env_vars_resources)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzserver_headless_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(clock_bridge)

    return ld
