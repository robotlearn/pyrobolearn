#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distribute several e-pucks in the world and make them move forward.

You can move in the world using the keyboard and mouse:
- `ctrl + left click`: rotate the camera
- `scroll wheel` or `ctrl + right click`: zoom in/out
- `ctrl + middle click`: move the camera
- `left click` on an object: if the object has a mass and a collision shape, you can interact with it with the mouse
- `w`: wireframe (see collision shapes)
- `g`: show/hide menu
- `esc`: quit the simulator
"""

import numpy as np
from itertools import count
import argparse

import pyrobolearn as prl


# create function for the parser to check the number of robots
def check(number):
    """check that the number of robots is between 1 and 100."""
    number = int(number)
    if number < 1:
        number = 1
    if number > 100:
        number = 100
    return number


# create parser to select the robot
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--number', help='the number of epucks in the world', type=check, default=10)
args = parser.parse_args()


# create simulator
sim = prl.simulators.Bullet()

# create basic world (with a floor and gravity enabled by default)
world = prl.worlds.BasicWorld(sim, scaling=1)

# specify distribution ranges for position (x,y,z) and orientation (r,p,y)
low_position, high_position = [-3, -3, 0], [3, 3, 0]  # x,y,z
low_orientation, high_orientation = [0, 0, -np.pi], [0, 0, np.pi]  # r,p,y

# distribute the epucks in the world
robots = world.distribute(world.load_robot, size=args.number, position_range=(low_position, high_position),
                          rpy_range=(low_orientation, high_orientation), return_body=True, robot='epuck')

# run simulator
for t in count():

    # move each robot forward
    for robot in robots:
        robot.drive(speed=5)

    # perform one step in the world
    world.step(sleep_dt=1. / 240)
