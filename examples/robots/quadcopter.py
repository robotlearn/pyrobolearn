#!/usr/bin/env python
"""Load the Quadcopter robotic platform.
"""

import numpy as np
from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Quadcopter

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Quadcopter(sim)

# print information about the robot
robot.printRobotInfo()

rpm = robot.getStationaryRPM()
print("Stationary RPM: {}".format(rpm))
v = robot.rpmToRadPerSecond(rpm+20)
v = [v, -v, v, -v]

# run simulation
for i in count():
    robot.setJointVelocities(v)
    # step in simulation
    world.step(sleep_dt=1./240)
