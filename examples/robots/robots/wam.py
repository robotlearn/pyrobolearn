# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Load the WAM robotic platform.
"""

import numpy as np
from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import WAM

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = WAM(sim)

# print information about the robot
robot.print_info()
# H = robot.get_mass_matrix()
# print("Inertia matrix: H(q) = {}".format(H))

robot.set_joint_positions([np.pi / 4, np.pi / 2], joint_ids=[0, 1]) #2, 4])

Jlin = robot.get_jacobian(6)[:3]
robot.draw_velocity_manipulability_ellipsoid(6, Jlin, color=(1, 0, 0, 0.7))
for _ in range(5):
    world.step(sleep_dt=1./240)

Jlin = robot.get_jacobian(6)[:3]
robot.draw_velocity_manipulability_ellipsoid(6, Jlin, color=(0, 0, 1, 0.7))
for _ in range(45):
    world.step(sleep_dt=1./240)

Jlin = robot.get_jacobian(6)[:3]
robot.draw_velocity_manipulability_ellipsoid(6, Jlin)

for i in count():
    if i%1000 == 0:
        print("Joint Torques: {}".format(robot.get_joint_torques()))
        print("Gravity Torques: {}".format(robot.get_gravity_compensation_torques()))
        print("Compensation Torques: {}".format(robot.get_coriolis_and_gravity_compensation_torques()))
    # step in simulation
    world.step(sleep_dt=1./240)
