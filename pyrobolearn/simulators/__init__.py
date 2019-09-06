
# load middlewares
from . import middlewares


# load all simulators

# basic simulator
from .simulator import Simulator

# Bullet simulator
from .bullet import Bullet

# Bullet ros simulator
from .bullet_ros import BulletROS

# Dart simulator
# from .dart import Dart

# MuJoCo simulator
# from .mujoco import Mujoco

# Raisim simulator
# from .raisim import Raisim

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
