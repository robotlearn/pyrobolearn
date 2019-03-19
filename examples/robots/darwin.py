#!/usr/bin/env python
"""Load the Darwin robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Darwin

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Darwin(sim, useFixedBase=False)

# print information about the robot
robot.printRobotInfo()
print(robot.link_names)

# Position control using sliders
# robot.addJointSlider()

# run simulator
for _ in count():
    # robot.updateJointSlider()
    world.step(sleep_dt=1./240)
