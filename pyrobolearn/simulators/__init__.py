# -*- coding: utf-8 -*-

# load middlewares
from . import middlewares

# check Python version
import sys
python_version = sys.version_info[0]

# load all simulators

# basic simulator
from .simulator import Simulator

# Bullet simulator
from .bullet import Bullet

# Bullet ros simulator
from .bullet_ros import BulletROS

if python_version >= 3:
    # Dart simulator
    try:
        from .dart import Dart
    except ImportError as e:
        print("Dart not found.")

    # MuJoCo simulator
    try:
        from .mujoco import Mujoco
    except ImportError as e:
        print("MuJoCo not found.")

# Raisim simulator
try:
    from .raisim import Raisim
except ImportError as e:
    print("Raisim not found.")

# Vrep simulator (note that there is a currently a problem when loading pybullet with pyrep)
# from .vrep import VREP

# Isaac simulator
# from .isaac import Isaac

# Gazebo simulator
# from .gazebo import Gazebo


# # PyBullet simulator
# import pybullet
# import pybullet_data
# from pybullet_envs.bullet.bullet_client import BulletClient
#
#
# def BulletSim(mode=pybullet.GUI, debug_visualizer=False):
#     """mode: pybullet.GUI, pybullet.DIRECT"""
#     sim = BulletClient(connection_mode=mode)
#     sim.setAdditionalSearchPath(pybullet_data.getDataPath())
#     if not debug_visualizer:
#         sim.configureDebugVisualizer(sim.COV_ENABLE_GUI, 0)
#     return sim
