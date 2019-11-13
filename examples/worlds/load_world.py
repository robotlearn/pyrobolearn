#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load a basic world with different objects in it

Load a basic world (i.e. with a floor and gravity enabled) with different objects (only visual, and with collision
shapes) that are movable, fixed, or are moving. The objects with a positive mass and a collision shape can be moved
in the simulator using the mouse.

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
import numpy as np

# import pyrobolearn
import pyrobolearn as prl
from pyrobolearn.utils.transformation import get_quaternion


# create simulator
sim = prl.simulators.Bullet()

# create basic world (with a floor and gravity enabled by default)
world = prl.worlds.BasicWorld(sim)


# load basic shapes (only visual and with collisions) #

# load a visual sphere (without collision shape)
# By setting `return_body` to True, it returns an instance of `Body` which have several methods
# and attributes, and will allow us to move the robot by setting its position
sphere = world.load_visual_sphere(position=[1., 0, 1.], radius=0.5, color=(1, 0, 0, 0.5), return_body=True)

# load a movable cylinder (with collision shape)
cylinder_id = world.load_cylinder(position=[0, -1, 1], color=(1, 0, 0, 1))

# load a movable box (with collision shape)
box_id = world.load_box(position=[-1, 0, 1], dimensions=[1., 1., 1.], color=(0, 0, 1, 1))

# load an non-movable ellipsoid (with collision shape). To make an object non-movable, set its mass to 0.
ellipsoid_id = world.load_ellipsoid(position=[0, 0, 2], mass=0, scale=[2., 1., 1.], color=(1, 1, 0, 1))

# load a movable capsule (with collision shape)
capsule_id = world.load_capsule(position=[0, 1, 1], orientation=get_quaternion([0, np.deg2rad(10), 0]),
                                color=[0, 1, 1, 1])

# load a visual cone (without collision shape)
cone_id = world.load_visual_cone(position=[1, 0, 0.5], orientation=(0, 1, 0, 0), scale=(1., 1., 1.),
                                 color=(1, 1, 1, 0.8))

# load a table via its urdf (the urdf is in the `pybullet_data`)
table_id = world.load_urdf('table/table.urdf', position=[2, 1.5, 0])

# run simulator
red = True
for t in count():

    # move the visual sphere
    sphere.position = np.array([np.cos(0.006283 * t), np.sin(0.006283 * t), 1.])

    # change the color of the sphere after one complete revolution
    if t % 1000 == 0:
        if red:
            world.change_body_color(sphere.id, (1, 0, 0, 0.5))  # red
        else:
            world.change_body_color(sphere.id, (0, 0, 1, 0.5))  # blue
        red = not red

    # perform one step in the world
    world.step(sleep_dt=1. / 240)
