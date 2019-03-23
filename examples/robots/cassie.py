#!/usr/bin/env python
"""Provide the Cassie robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Cassie

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Cassie(sim)

# print information about the robot
robot.print_info()

# position control using sliders
# robot.add_joint_slider()

# run simulator
for _ in count():
    # robot.update_joint_slider()
    robot.move_joint_home_positions()
    world.step(sleep_dt=1./240)
