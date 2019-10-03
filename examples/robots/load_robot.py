# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Load a robot in a basic world.

You can move in the world using the keyboard and mouse:
- `ctrl + left click`: rotate the camera
- `scroll wheel` or `ctrl + right click`: zoom in/out
- `ctrl + middle click`: move the camera
- `left click` on an object: if the object has a mass and a collision shape, you can interact with it with the mouse
- `w`: wireframe (see collision shapes)
- `g`: show/hide menu
- `esc`: quit the simulator
"""

from itertools import count
import argparse

import pyrobolearn as prl


# get implemented robots
robots = prl.robots.implemented_robots
print("All the robots (total number of robots = {}): {}".format(len(robots), robots))

# create parser to select the robot
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', help='the robot to load in the world', type=str,
                    choices=robots, default='hyq2max')
args = parser.parse_args()


# create simulator
sim = prl.simulators.Bullet()

# create basic world with floor and gravity
world = prl.worlds.BasicWorld(sim)

# load the robot in the world (note that you can create the robot outside the world (not recommended),
# and then give it to the `world.load_robot` method to let know the world that a robot was loaded)
robot = world.load_robot(robot=args.robot, position=[0., 0.])

# run simulator
for _ in count():
    # perform one step in the world
    world.step(sleep_dt=1. / 240)
