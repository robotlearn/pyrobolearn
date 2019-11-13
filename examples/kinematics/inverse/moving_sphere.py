#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inverse kinematics with the Kuka robot where the goal is to follow a moving sphere.
"""

import numpy as np
from itertools import count

from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA, Body


# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = KukaIIWA(sim)
robot.print_info()
world.load_robot(robot)

# define useful variables for IK
dt = 1./240
link_id = robot.get_end_effector_ids(end_effector=0)
joint_ids = robot.joints  # actuated joint
damping = 0.01  # for damped-least-squares IK
wrt_link_id = -1  # robot.get_link_ids('iiwa_link_1')
qIdx = robot.get_q_indices(joint_ids)

# define gains
kp = 50    # 5 if velocity control, 50 if position control
kd = 0     # 2*np.sqrt(kp)

# create sphere to follow
sphere = world.load_visual_sphere(position=np.array([0.5, 0., 1.]), radius=0.05, color=(1, 0, 0, 0.5))
sphere = Body(sim, body_id=sphere)

# set initial joint positions (based on the position of the sphere at [0.5, 0, 1])
robot.reset_joint_states(q=[8.84305270e-05, 7.11378917e-02, -1.68059886e-04, -9.71690439e-01, 1.68308810e-05,
                            3.71467111e-01, 5.62890805e-05])

# define amplitude and angular velocity when moving the sphere
w = 0.01
r = 0.2

for t in count():
    # move sphere
    sphere.position = np.array([0.5, r * np.cos(w*t + np.pi/2), (1.-r) + r * np.sin(w*t + np.pi/2)])

    # get current end-effector position and velocity in the task/operational space
    x = robot.get_link_world_positions(link_id)
    dx = robot.get_link_world_linear_velocities(link_id)

    # Get joint positions
    q = robot.get_joint_positions()

    # Get linear jacobian
    if robot.has_floating_base():
        J = robot.get_linear_jacobian(link_id, q=q)[:, qIdx + 6]
    else:
        J = robot.get_linear_jacobian(link_id, q=q)[:, qIdx]

    # Pseudo-inverse: \hat{J} = J^T (JJ^T + k^2 I)^{-1}
    Jp = robot.get_damped_least_squares_inverse(J, damping)

    # evaluate damped-least-squares IK
    dq = Jp.dot(kp * (sphere.position - x) - kd * dx)

    # set joint positions
    q = q[qIdx] + dq * dt
    robot.set_joint_positions(q, joint_ids=joint_ids)

    # step in simulation
    world.step(sleep_dt=dt)
