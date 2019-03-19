#!/usr/bin/env python
"""Load the Ant Mujoco model.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Ant

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Ant(sim)  # , useFixedBase=True)

# print information about the robot
robot.printRobotInfo()

# Position control using sliders
# robot.addJointSlider(robot.getLeftFrontLegIds() + robot.getRightFrontLegIds())

# run simulator
for _ in count():
    # robot.updateJointSlider()
    world.step(sleep_dt=1./240)
