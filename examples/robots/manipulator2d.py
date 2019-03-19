#!/usr/bin/env python
"""Load 2d manipulators.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Manipulator2D

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Manipulator2D(sim, init_pos=(0, -0.25, 0))
robot1 = Manipulator2D(sim, init_pos=(0, 0.25, 0))
robot.printRobotInfo()

# Position control using sliders
# robot.addJointSlider()

# run simulator
for _ in count():
    # robot.updateJointSlider()
    world.step(sleep_dt=1./240)
