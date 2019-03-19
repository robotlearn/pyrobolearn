#!/usr/bin/env python
"""Load the WAM robotic platform.
"""

import numpy as np
from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import WAM

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = WAM(sim)

# print information about the robot
robot.printRobotInfo()
# H = robot.calculateMassMatrix()
# print("Inertia matrix: H(q) = {}".format(H))

robot.setJointPositions([np.pi / 4, np.pi / 2], jointId=[0,1]) #2, 4])

Jlin = robot.calculateJacobian(6, localPosition=(0., 0., 0.))[:3]
robot.drawVelocityManipulabilityEllipsoid(6, Jlin, color=(1,0,0,0.7))
for _ in range(5):
    world.step(sleep_dt=1./240)

Jlin = robot.calculateJacobian(6, localPosition=(0., 0., 0.))[:3]
robot.drawVelocityManipulabilityEllipsoid(6, Jlin, color=(0, 0, 1, 0.7))
for _ in range(45):
    world.step(sleep_dt=1./240)

Jlin = robot.calculateJacobian(6, localPosition=(0., 0., 0.))[:3]
robot.drawVelocityManipulabilityEllipsoid(6, Jlin)

for i in count():
    if i%1000 == 0:
        print("Joint Torques: {}".format(robot.getJointTorques()))
        print("Gravity Torques: {}".format(robot.getGravityCompensationTorques()))
        print("Compensation Torques: {}".format(robot.getCoriolisAndGravityCompensationTorques()))
    # step in simulation
    world.step(sleep_dt=1./240)
