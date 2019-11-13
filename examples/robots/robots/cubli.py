#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load the Cubli robot.
"""

import numpy as np
from itertools import count
from pyrobolearn.utils.transformation import get_rpy_from_quaternion
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Cubli

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
scale = 1.  # Warning: this does not scale the mass...
position = [0., 0., np.sqrt(2) / 2. * scale + 0.001]
orientation = [0.383, 0, 0, 0.924]
robot = Cubli(sim, position, orientation, scale=scale)

# print information about the robot
robot.print_info()
H = robot.get_mass_matrix(q_idx=slice(6, 6 + len(robot.joints)))  # floating base, thus keep only the last q
print("Inertia matrix: H(q) = {}\n".format(H))

# PD control
Kp = 600.
Kd = 2 * np.sqrt(Kp)
desired_roll = np.pi / 4.

for i in count():
    # get state
    quaternion = robot.get_base_orientation()
    w = robot.get_base_angular_velocity()
    euler = get_rpy_from_quaternion(quaternion)

    # PD control
    torques = [-Kp * (desired_roll - euler[0]) + Kd * w[0], 0., 0.]
    robot.set_joint_torques(torques)

    # step in simulation
    world.step(sleep_dt=1./240)
