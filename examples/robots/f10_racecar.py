#!/usr/bin/env python
"""Load the F10 racecar robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import F10Racecar

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = F10Racecar(sim)

# print information about the robot
robot.printRobotInfo()

# Position control using sliders
# robot.addJointSlider()

# run simulator
for _ in count():
    # robot.updateJointSlider()
    robot.driveForward(10)
    world.step(sleep_dt=1./240)
