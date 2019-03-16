
# load all simulators

# basic simulator
from simulator import Simulator

# PyBullet simulator
import pybullet
import pybullet_data
from pybullet_envs.bullet.bullet_client import BulletClient


def BulletSim(mode=pybullet.GUI, debug_visualizer=False):
    """mode: pybullet.GUI, pybullet.DIRECT"""
    sim = BulletClient(connection_mode=mode)
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    if not debug_visualizer:
        sim.configureDebugVisualizer(sim.COV_ENABLE_GUI, 0)
    return sim
