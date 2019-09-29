# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the HyQ2Max robot.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import HyQ2Max

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)
world.load_japanese_monastery()

# create robot
robot = HyQ2Max(sim)

# print information about the robot
robot.print_info()

# Position control using sliders
# robot.add_joint_slider(robot.getLeftFrontLegIds())

# run simulator
for _ in count():
    # robot.update_joint_slider()
    robot.compute_and_draw_com_position()
    robot.compute_and_draw_projected_com_position()

    world.step(sleep_dt=1./240)
