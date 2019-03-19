#!/usr/bin/env python
"""Load the Humanoid Mujoco model.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Humanoid

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Humanoid(sim)

# print information about the robot
robot.printRobotInfo()

# Position control using sliders
# robot.addJointSlider()

# run simulation
for i in count():
    # robot.updateJointSlider()
    # step in simulation
    world.step(sleep_dt=1./240)
