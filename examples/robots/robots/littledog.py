#!/usr/bin/env python
"""Load the Little Dog robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import LittleDog

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = LittleDog(sim)

# print information about the robot
robot.print_info()

# Position control using sliders
# robot.add_joint_slider()

# run simulator
for _ in count():
    # robot.update_joint_slider()
    # robot.compute_and_draw_com_position()
    # robot.compute_and_draw_projected_com_position()
    world.step(sleep_dt=1./240)
