from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rl_caging_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'models', 'cylindrical_object'), glob('models/cylindrical_object/*')),
        (os.path.join('share', package_name, 'models', 'turtlebot3_burger'), glob('models/turtlebot3_burger/*')),
        (os.path.join('share', package_name, 'models', 'diff_drive_bot'), glob('models/diff_drive_bot/*')),
        (os.path.join('share', package_name, 'models', 'target_flag'), glob('models/target_flag/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Thaddeus Rosales',
    maintainer_email='thadyboygwapo82@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'random = rl_caging_simulation.random:main',
            'read_state = rl_caging_simulation.read_state:main',
        ],
    },
)
