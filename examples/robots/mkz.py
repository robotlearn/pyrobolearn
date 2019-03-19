#!/usr/bin/env python
"""Load the Lincoln MKZ car robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import MKZ

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = MKZ(sim)

# print information about the robot
robot.printRobotInfo()

# Position control using sliders
# robot.addJointSlider()

# run simulator
for _ in count():
    # robot.updateJointSlider()
    robot.driveForward(2)
    world.step(sleep_dt=1./240)
