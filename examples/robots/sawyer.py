#!/usr/bin/env python
"""Load the Sawyer robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Sawyer

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Sawyer(sim)

# print information about the robot
robot.printRobotInfo()

# # Position control using sliders
robot.addJointSlider()

# run simulator
for _ in count():
    robot.updateJointSlider()
    world.step(sleep_dt=1./240)
