#!/usr/bin/env python
"""Load the Coman robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Coman

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Coman(sim, fixed_base=True)

# print information about the robot
robot.print_info()
print(robot.link_names)

# # Position control using sliders
# robot.add_joint_slider()

robot.change_transparency()
# robot.draw_link_coms()
robot.draw_link_frames()
# robot.draw_bounding_boxes(robot.right_leg[4])

# run simulator
for _ in count():
    # robot.update_joint_slider()
    # robot.compute_and_draw_com_position()
    # robot.compute_and_draw_projected_com_position()
    world.step(sleep_dt=1./240)
