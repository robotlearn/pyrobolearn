#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Draw the 2D velocity and force manipulability ellipsoids on the end-effector of a 3-link planar manipulator.

References:
    [1] "Robotics: Modelling, Planning and Control" (section 3.9), Siciliano et al., 2010
"""

import time
# from itertools import count
import numpy as np

import pyrobolearn as prl


# create simulator
sim = prl.simulators.Bullet()

# create world
world = prl.worlds.BasicWorld(sim)

# create robot
robot = world.load_robot('manipulator2d')
robot.reset_joint_states(q=[0.64453457, -1.65045902, -0.31141744])

# change camera view
world.camera.reset(distance=2, yaw=-np.pi / 2, pitch=-np.pi/2.01)

# draw 2d velocity manipulability ellipsoid
# print(robot.end_effector_names)
end_effector_id = robot.get_link_ids('gripper')
jacobian = robot.get_linear_jacobian(link_id=end_effector_id)
jjt = robot.get_JJT(jacobian)
robot.draw_velocity_manipulability_ellipsoid(link_id=end_effector_id, JJT=jjt, color=(0, 1, 0, 0.7))  # green
robot.draw_force_manipulability_ellipsoid(link_id=end_effector_id, JJT=jjt, color=(1, 0, 0, 0.7))     # red

# TODO: fix bug

time.sleep(10000)
# run simulator
# for t in count():
#     world.step(sleep_dt=1./240)
