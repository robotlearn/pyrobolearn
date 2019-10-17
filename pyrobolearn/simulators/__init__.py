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
# from .bullet_ros import BulletROS  # Deprecated

if python_version >= 3:
    # Dart simulator
    try:
        import dartpy
        from .dart import Dart
    except ImportError as e:
        print("Dart could not be found on this system... Skipping prl.simulators.Dart...")

    # MuJoCo simulator
    try:
        import mujoco_py
        from .mujoco import Mujoco
    except ImportError as e:
        print("MuJoCo could not be found on this system... Skipping prl.simulators.Mujoco...")

# Raisim simulator
try:
    import raisimpy
    from .raisim import Raisim
except ImportError as e:
    print("Raisim could not be found on this system... Skipping prl.simulators.Raisim...")

# # Vrep simulator (note that there is a currently a problem when loading pybullet with pyrep)
# try:
#     import pyrep
#     from .vrep import VREP
# except ImportError as e:
#     print("V-REP not found.")

# # Isaac simulator
# try:
#     import isaacgym
#     from .isaac import Isaac
# except ImportError as e:
#     print("Isaac not found.")

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
