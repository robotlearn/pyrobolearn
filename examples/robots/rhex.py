#!/usr/bin/env python
"""Load the Rhex robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Rhex

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Rhex(sim)

# print information about the robot
robot.printRobotInfo()

# Position control using sliders
# robot.addJointSlider(robot.right_back_leg)

# run simulation
for i in count():
    # robot.updateJointSlider()
    robot.drive(2)
    # step in simulation
    world.step(sleep_dt=1./240)
