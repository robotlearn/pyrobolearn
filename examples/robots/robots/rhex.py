#!/usr/bin/env python
"""Load the Rhex robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Rhex

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = Rhex(sim)

# print information about the robot
robot.print_info()

# Position control using sliders
# robot.add_joint_slider(robot.right_back_leg)

# run simulation
for i in count():
    # robot.update_joint_slider()
    robot.drive(2)
    # step in simulation
    world.step(sleep_dt=1./240)
