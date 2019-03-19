#!/usr/bin/env python
"""Load the Laikago robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Laikago

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Laikago(sim)

# print information about the robot
robot.printRobotInfo()

# # Position control using sliders
# robot.addJointSlider()

# run simulator
for _ in count():
    # robot.updateJointSlider()
    robot.moveJointHomePositions()
    world.step(sleep_dt=1./240)
