#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load some Epuck robots.
"""

import numpy as np
from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Epuck

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robots = []
for _ in range(5):
    x, y = np.random.uniform(low=-2, high=2, size=2)
    robot = world.load_robot(Epuck, position=(x, y, 0))
    robots.append(robot)

# print information about the robot
robots[0].print_info()

# Position control using sliders
# robots[0].add_joint_slider()

# run simulator
for _ in count():
    # robots[0].update_joint_slider()
    for robot in robots:
        robot.drive(5)
    world.step(sleep_dt=1./240)
