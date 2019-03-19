#!/usr/bin/env python
"""Load the Crab robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Crab

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Crab(sim)

# print information about the robot
robot.printRobotInfo()

# Position control using sliders
robot.addJointSlider(robot.right_middle_leg)

# run simulation
for i in count():
    robot.updateJointSlider()
    # step in simulation
    world.step(sleep_dt=1./240)
