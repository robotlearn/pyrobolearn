# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Load the Pepper robot.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Pepper


# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = Pepper(sim)

# print information about the robot
robot.print_info()

# Position control using sliders
# robot.add_joint_slider()

# run simulator
for i in count():
    # robot.update_joint_slider()
    if i % 20 == 0:
        robot.camera_top.get_rgb_image()
    world.step(sleep_dt=1./240)
