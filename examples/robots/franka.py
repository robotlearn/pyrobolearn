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
robot.printRobotInfo()
# H = robot.calculateMassMatrix()
# print("Inertia matrix: H(q) = {}".format(H))

# Position control using sliders
# robot.addJointSlider()

for i in count():
    # robot.updateJointSlider()
    # step in simulation
    world.step(sleep_dt=1./240)
