#!/usr/bin/env python
"""Load the Pepper robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Pepper


# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Pepper(sim)

# print information about the robot
robot.printRobotInfo()

# Position control using sliders
# robot.addJointSlider()

# run simulator
for i in count():
    # robot.updateJointSlider()
    if i % 20 == 0:
        robot.cameraTop.getRGBImage()
    world.step(sleep_dt=1./240)
