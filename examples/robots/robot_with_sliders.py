# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Manipulate the robot's joints with sliders.

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


# create parser to select the robot
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', help='the robot to load in the world', type=str,
                    choices=prl.robots.implemented_robots, default='coman')
parser.add_argument('-f', '--fixed_base', help='if we should fix the base when the robot has initially a floating '
                                               'base', type=bool, default=True)
args = parser.parse_args()


# create simulator
sim = prl.simulators.Bullet()

# create basic world with floor and gravity
world = prl.worlds.BasicWorld(sim)

# load the robot in the world
robot = world.load_robot(robot=args.robot, position=[0., 0.], fixed_base=args.fixed_base)

# add a slider for each specified joint
robot.add_joint_slider(joint_ids=robot.joints)

# run simulator
for _ in count():
    # update the joint slider
    robot.update_joint_slider()

    # perform one step in the world
    world.step(sleep_dt=1. / 240)
