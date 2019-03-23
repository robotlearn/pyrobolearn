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
world.load_sphere([2., 0, 2.], mass=0., color=(1, 0, 0, 1))
world.load_sphere([2., 1., 2.], mass=0., color=(0, 0, 1, 1))

# load robot
robot = Walkman(sim, fixed_base=False, lower_body=False)

# print information about the robot
robot.print_info()

# # Position control using sliders
robot.add_joint_slider(robot.left_leg)

# run simulator
for i in count():
    robot.update_joint_slider()
    if i % 60 == 0:
        img = robot.left_camera.get_rgb_image()

    world.step(sleep_dt=1./240)
