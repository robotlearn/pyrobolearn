#!/usr/bin/env python
"""Load the Cogimon robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Cogimon

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Cogimon(sim, lower_body=False)

# print information about the robot
robot.printRobotInfo()

# # Position control using sliders
robot.addJointSlider(robot.left_leg)

# run simulator
for _ in count():
    robot.updateJointSlider()
    world.step(sleep_dt=1./240)
