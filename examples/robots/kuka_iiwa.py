#!/usr/bin/env python
"""Provide the Kuka IIWA robotic platform.
"""

import numpy as np
from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = KukaIIWA(sim)

# print information about the robot
robot.print_info()
# H = robot.get_mass_matrix()
# print("Inertia matrix: H(q) = {}".format(H))

# print(robot.get_link_world_positions(flatten=False))

K = 5000*np.identity(3)
# D = 2 * np.sqrt(K)
# D = np.zeros((3,3))
D = 100 * np.identity(3)
x_des = np.array([0.3, 0.0, 0.8])
x_des = np.array([0.52557296, 0.09732758, 0.80817658])
link_id = robot.get_link_ids('iiwa_link_ee')

for i in count():
    # print(robot.get_link_world_positions(flatten=False))

    # get state
    q = robot.get_joint_positions()
    dq = robot.get_joint_velocities()
    x = robot.get_link_world_positions(link_id)
    dx = robot.get_link_world_linear_velocities(link_id)

    # get (linear) jacobian
    J = robot.get_linear_jacobian(link_id, q)

    # get coriolis, gravity compensation torques
    torques = robot.get_coriolis_and_gravity_compensation_torques(q, dq)

    # Impedance control: attractor point
    F = K.dot(x_des - x) - D.dot(dx)
    # F = -D.dot(dx)
    tau = J.T.dot(F)
    print(tau)
    torques += tau
    robot.set_joint_torques(torques)

    # step in simulation
    world.step(sleep_dt=1./240)
