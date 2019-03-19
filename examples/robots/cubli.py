#!/usr/bin/env python
"""Load the Cubli robot.
"""

import numpy as np
from itertools import count
from pyrobolearn.simulators import BulletSim, pybullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Cubli

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
scale = 1.  # Warning: this does not scale the mass...
position = [0., 0., np.sqrt(2) / 2. * scale + 0.001]
orientation = [0.383, 0, 0, 0.924]
robot = Cubli(sim, position, orientation, scaling=scale)

# print information about the robot
robot.printRobotInfo()
H = robot.calculateMassMatrix(qIdx=slice(6, 6+len(robot.joints)))  # floating base, thus keep only the last q
print("Inertia matrix: H(q) = {}\n".format(H))

# PD control
Kp = 600.
Kd = 2 * np.sqrt(Kp)
desired_roll = np.pi / 4.

for i in count():
    # get state
    quaternion = robot.getBaseOrientation(False)
    w = robot.getBaseAngularVelocity()
    euler = pybullet.getEulerFromQuaternion(quaternion.tolist())

    # PD control
    torques = [-Kp * (desired_roll - euler[0]) + Kd * w[0], 0., 0.]
    robot.setJointTorques(torques)

    # step in simulation
    world.step(sleep_dt=1./240)
