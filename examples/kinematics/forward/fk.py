#!/usr/bin/env python
"""In this file, we perform forward kinematics using the Kuka robot.

The Kuka robot just draw a circle in the air. The joint positions are in the `data.txt` file.
"""

import os
import pickle
from itertools import count

from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA, Body


# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = KukaIIWA(sim)
robot.print_info()

# define useful variables for FK
link_id = robot.get_end_effector_ids(end_effector=0)
joint_ids = robot.joints  # actuated joint

# load data
with open(os.path.dirname(os.path.abspath(__file__)) + '/data.txt', 'rb') as f:
    positions = pickle.load(f)

# set initial joint position
robot.reset_joint_states(q=positions[0])

# draw a sphere at the position of the end-effector
sphere = world.load_visual_sphere(position=robot.get_link_world_positions(link_id),
                                  radius=0.05, color=(1, 0, 0, 0.5))
sphere = Body(sim, body_id=sphere)

# perform simulation
for t in count():

    # if no more joint positions, get out of the loop
    if t >= len(positions):
        break

    # set joint positions
    robot.set_joint_positions(positions[t], joint_ids=joint_ids)

    # make the sphere follow the end effector
    sphere.position = robot.get_link_world_positions(link_id)

    # step in simulation
    world.step(sleep_dt=1./240)
