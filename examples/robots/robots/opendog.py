#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load the OpenDog robot.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import OpenDog

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = OpenDog(sim)

# print information about the robot
robot.print_info()

# Position control using sliders
# robot.add_joint_slider(robot.getLeftFrontLegIds())

# run simulator
for _ in count():
    # robot.update_joint_slider()
    world.step(sleep_dt=1./240)
