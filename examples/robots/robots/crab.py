#!/usr/bin/env python
"""Load the Crab robot.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Crab

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = Crab(sim)

# print information about the robot
robot.print_info()

# Position control using sliders
robot.add_joint_slider(robot.right_middle_leg)

# run simulation
for i in count():
    robot.update_joint_slider()
    # step in simulation
    world.step(sleep_dt=1./240)
