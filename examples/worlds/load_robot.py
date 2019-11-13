#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""load a robot in a basic world and distribute randomly few objects on the floor.

Load a basic world (i.e. with a floor and gravity enabled), and load a robot inside of it, and distributed randomly
some small cubes in front of it.

You can move in the world using the keyboard and mouse:
- `ctrl + left click`: rotate the camera
- `scroll wheel` or `ctrl + right click`: zoom in/out
- `ctrl + middle click`: move the camera
- `left click` on an object: if the object has a mass and a collision shape, you can interact with it with the mouse
- `w`: wireframe (see collision shapes)
- `g`: show/hide menu
- `esc`: quit the simulator
"""

# import standard libraries
from itertools import count
import argparse

# import pyrobolearn
import pyrobolearn as prl


# create parser to select the robot
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', help='the robot to load in the world', type=str,
                    choices=prl.robots.implemented_robots, default='hyq2max')
args = parser.parse_args()


# create simulator
sim = prl.simulators.Bullet()

# create basic world (with a floor and gravity enabled by default)
world = prl.worlds.BasicWorld(sim)

# load the robot in the world (note that you can create the robot outside the world (not recommended),
# and then give it to the `world.load_robot` method to let know the world that a robot was loaded)
robot = world.load_robot(robot=args.robot, position=[0., 0.])

# distribute some boxes in the world
# There are 2 ways to carry this out.
# 1. create one instance first (the position is decided by the user), and then distribute
box_id = world.load_box(position=[2., 0, 0.05], mass=0.1, dimensions=[0.05, 0.05, 0.05], color=(1, 0, 0, 1))
box_ids1 = world.distribute(box_id, size=10, position_range=([1, -1, 0.05], [3, 1, 0.05]))
# 2. directly distribute by passing the function as an argument to `world.distribute`
box_ids2 = world.distribute(world.load_box, size=10, position_range=([-1, -3, 0.05], [1, -1, 0.05]), mass=0.1,
                            dimensions=[0.05, 0.05, 0.05], color=(0, 1, 0, 1))

# run simulator
for t in count():
    # perform one step in the world
    world.step(sleep_dt=1. / 240)
