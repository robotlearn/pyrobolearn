#!/usr/bin/env python
"""Load the Ballbot robot.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Ballbot

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = Ballbot(sim)

# print information about the robot
robot.print_info()

for i in count():
    # robot.set_joint_velocities([0, -1, 0])

    # step in simulation
    world.step(sleep_dt=1./240)
