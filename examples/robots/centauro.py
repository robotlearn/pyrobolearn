#!/usr/bin/env python
"""Load the Centauro robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Centauro

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# load robot
robot = Centauro(sim)  # , useFixedBase=True)

# print information about the robot
robot.printRobotInfo()
print("Number of Legs: {}".format(robot.getNumberOfLegs()))
print("Number of Arms: {}".format(robot.getNumberOfArms()))

# robot.addJointSlider(robot.getRightFrontLegIds())
robot.drive(speed=3)

# run simulator
for _ in count():
    robot.updateJointSlider()
    world.step(sleep_dt=1./240)
