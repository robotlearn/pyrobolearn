#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Attractor point using impedance control with RRBot

Try to move the end-effector using the mouse, and see what happens. Compare the obtained results with
`force/no_forces.py` and `force/gravity_compensation.py`.
"""

import numpy as np
from itertools import count

import pyrobolearn as prl


# Create simulator
sim = prl.simulators.Bullet()

# create world
world = prl.worlds.BasicWorld(sim)

# load robot
robot = prl.robots.RRBot(sim)
robot.disable_motor()           # disable motors; comment the `robot.set_joint_torques(torques)` to see what happens
robot.print_info()
robot.change_transparency()
world.load_robot(robot)

# define variables
link_id = robot.get_link_ids('hokuyo_link')         # the link we are interested to
com_frame = robot.get_link_states(link_id)[2]
x_des = robot.get_link_world_positions(link_id)     # desired cartesian position

# gains
K = 100 * np.identity(3)
D = 2 * np.sqrt(K)  # critically damped
D = 3 * D           # manually increase damping

# draw a sphere at the desired location
world.load_visual_sphere(position=x_des, radius=0.1, color=(0, 1, 0, 0.5))


# run simulator
for _ in count():
    # get current joint positions, velocities, accelerations
    q = robot.get_joint_positions()
    dq = robot.get_joint_velocities()
    ddq = np.zeros(len(q))

    # get current link position and velocity
    x = robot.get_link_world_positions(link_id)
    dx = robot.get_link_world_linear_velocities(link_id)

    # compute torques (Coriolis, centrifugal and gravity compensation) using inverse dynamics
    torques = robot.calculate_inverse_dynamics(ddq, dq, q)

    # get linear jacobian
    Jlin = robot.get_linear_jacobian(link_id=link_id, local_position=com_frame)

    # attractor point: compute cartesian forces (PD control)
    F = K.dot(x_des - x) - D.dot(dx)

    # add torques resulting from them
    torques += Jlin.T.dot(F)
    # torques += Jlin.T.dot(- D.dot(dx))  # active compliance
    # torques = Jlin.T.dot(F)

    # impedance control
    robot.set_joint_torques(torques=torques)

    # perform a step in the world
    world.step(sleep_dt=1./240)
