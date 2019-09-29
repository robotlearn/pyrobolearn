# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Load the Humanoid Mujoco model.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Humanoid

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = Humanoid(sim)

# print information about the robot
robot.print_info()

# Position control using sliders
# robot.add_joint_slider()

# run simulation
for i in count():
    # robot.update_joint_slider()
    # step in simulation
    world.step(sleep_dt=1./240)
