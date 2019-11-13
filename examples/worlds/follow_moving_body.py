#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Follow a body with the main camera in the world.

Try to move the sphere with the mouse (left-click on the object).
"""

from itertools import count

import pyrobolearn as prl

# create simulator
sim = prl.simulators.Bullet()

# create basic world (with a floor and gravity enabled by default)
world = prl.worlds.BasicWorld(sim)

# load sphere
sphere = world.load_sphere(position=[0, 0, 15.], radius=0.2, mass=1, color=(1, 0, 0, 1))

# run simulator
for _ in count():
    # follow sphere
    world.follow(sphere)
    # perform one step in the world
    world.step(sleep_dt=1. / 240)
