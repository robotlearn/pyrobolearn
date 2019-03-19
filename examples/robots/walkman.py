#!/usr/bin/env python
"""Load the WALK-MAN robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Walkman

# Create simulator
sim = BulletSim()

# Create world
world = BasicWorld(sim)
world.loadSphere([2., 0, 2.], mass=0., color=(1, 0, 0, 1))
world.loadSphere([2., 1., 2.], mass=0., color=(0, 0, 1, 1))

# load robot
robot = Walkman(sim, useFixedBase=False, lower_body=False)

# print information about the robot
robot.printRobotInfo()

# # Position control using sliders
robot.addJointSlider(robot.left_leg)

# run simulator
for i in count():
    robot.updateJointSlider()
    if i % 60 == 0:
        img = robot.left_camera.getRGBImage()

    world.step(sleep_dt=1./240)
