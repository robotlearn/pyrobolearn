#!/usr/bin/env python
"""Load the BB8 robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import BB8

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = BB8(sim)

# print information about the robot
robot.print_info()

for i in count():
    robot.set_joint_velocities([0, -1, 0])

    # step in simulation
    world.step(sleep_dt=1./240)
