#!/usr/bin/env python
"""Load 2d manipulators.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Manipulator2D

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = Manipulator2D(sim, position=(0, -0.25, 0))
robot1 = Manipulator2D(sim, position=(0, 0.25, 0))
robot.print_info()

# Position control using sliders
# robot.add_joint_slider()

# run simulator
for _ in count():
    # robot.update_joint_slider()
    world.step(sleep_dt=1./240)
