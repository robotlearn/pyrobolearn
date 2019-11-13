#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load the Lincoln MKZ car robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import MKZ

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = MKZ(sim)

# print information about the robot
robot.print_info()

# Position control using sliders
# robot.add_joint_slider()

# run simulator
for _ in count():
    # robot.update_joint_slider()
    robot.drive_forward(2)
    world.step(sleep_dt=1./240)
