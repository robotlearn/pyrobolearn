#!/usr/bin/env python
"""Load some Epuck robots.
"""

import numpy as np
from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Epuck

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robots = []
for _ in range(5):
    x, y = np.random.uniform(low=-2, high=2, size=2)
    robot = world.loadRobot(Epuck, position=(x, y, 0))
    robots.append(robot)

# print information about the robot
robots[0].printRobotInfo()

# Position control using sliders
# robots[0].addJointSlider()

# run simulator
for _ in count():
    # robots[0].updateJointSlider()
    for robot in robots:
        robot.drive(5)
    world.step(sleep_dt=1./240)
