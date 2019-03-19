#!/usr/bin/env python
"""Provide the HyQ2Max robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import HyQ2Max

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)
world.loadJapaneseMonastery()

# create robot
robot = HyQ2Max(sim)

# print information about the robot
robot.printRobotInfo()

# Position control using sliders
# robot.addJointSlider(robot.getLeftFrontLegIds())

# run simulator
for _ in count():
    # robot.updateJointSlider()
    robot.computeAndDrawCoMPosition()
    robot.computeAndDrawProjectedCoMPosition()

    world.step(sleep_dt=1./240)
