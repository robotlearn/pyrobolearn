#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test different visualization tools that can be used on the robot.


Test different visualization tools on a robot. You can notably:
- render the robot semi-transparent
- draw the robot center of mass and the projected center of mass
- draw the center of mass of each link
- draw the link frames
- draw the joint axis
- draw bounding boxes around links
- for legged robots:
    - draw ground reference points such as the ZMP, COP, FRI, and CMP
    - draw the support polygon
    - draw friction cones
- draw velocity and dynamic manipulability ellipsoids


You can move in the world using the keyboard and mouse:
- `ctrl + left click`: rotate the camera
- `scroll wheel` or `ctrl + right click`: zoom in/out
- `ctrl + middle click`: move the camera
- `left click` on an object: if the object has a mass and a collision shape, you can interact with it with the mouse
- `w`: wireframe (see collision shapes)
- `g`: show/hide menu
- `esc`: quit the simulator
"""

import time
from itertools import count
import argparse

import pyrobolearn as prl


# create parser to select the robot
robots = ['coman', 'hyq2max']  # prl.robots.implemented_robots
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', help='the robot to load in the world', type=str, choices=robots, default='hyq2max')
args = parser.parse_args()


# create simulator
sim = prl.simulators.Bullet()

# create basic world with floor and gravity
world = prl.worlds.BasicWorld(sim)

# load the robot in the world
robot = world.load_robot(robot=args.robot, position=[0., 0.])

# move the simulation a bit forward (such that the robot touches the floor)
for _ in range(100):
    world.step()

# change visualization
robot.change_transparency()
robot.draw_link_coms()
robot.draw_link_frames(robot.legs[0])
robot.draw_joint_frames(robot.legs[0])
robot.draw_bounding_boxes(link_ids=-1)

robot.draw_friction_cone(floor_id=world.floor_id)
robot.draw_support_polygon(floor_id=world.floor_id, lifetime=0)

robot.compute_and_draw_com_position()
robot.compute_and_draw_projected_com_position()
# robot.draw_cop(cop=world.floor_id)
# robot.draw_zmp(zmp=world.floor_id)
# robot.draw_cmp(cmp=world.floor_id)

time.sleep(10000)
# run simulator
# for t in count():
#     # perform one step in the world
#     world.step(sleep_dt=1. / 240)
