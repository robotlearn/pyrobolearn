#!/usr/bin/env python
"""Load the Franka Emika robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Franka

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Franka(sim)

# print information about the robot
robot.print_info()
# H = robot.get_mass_matrix()
# print("Inertia matrix: H(q) = {}".format(H))

# Position control using sliders
# robot.add_joint_slider()

for i in count():
    # robot.update_joint_slider()
    # step in simulation
    world.step(sleep_dt=1./240)
