# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Load the Centauro robot.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Centauro

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# load robot
robot = Centauro(sim)  # , fixed_base=True)

# print information about the robot
robot.print_info()
print("Number of Legs: {}".format(robot.num_legs))
print("Number of Arms: {}".format(robot.num_arms))

# robot.add_joint_slider(robot.getRightFrontLegIds())
robot.drive(speed=3)

# run simulator
for _ in count():
    robot.update_joint_slider()
    world.step(sleep_dt=1./240)
