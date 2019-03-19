#!/usr/bin/env python
"""Load the Atlas robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Atlas

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Atlas(sim)

# print information about the robot
robot.printRobotInfo()

# position control using sliders
robot.addJointSlider(robot.getLeftLegIds())

# run simulator
for _ in count():
    robot.updateJointSlider()
    world.step(sleep_dt=1./240)
