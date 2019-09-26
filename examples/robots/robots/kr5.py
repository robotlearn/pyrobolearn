# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Load the Kuka KR5 robotic industrial platform.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KR5

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = KR5(sim)

# print information about the robot
robot.print_info()
# H = robot.get_mass_matrix()
# print("Inertia matrix: H(q) = {}".format(H))

for i in count():
    # step in simulation
    world.step(sleep_dt=1./240)
