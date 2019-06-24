#!/usr/bin/env python
"""Example on how to use the Bullet simulator in pyrobolearn.

Simple example where we use the bullet simulator, load a basic world (with a floor and gravity enabled) and the RRBot
robot in it.

You can move inside the world using your mouse and keyboard:
- `ctrl + left click`: rotate the camera
- `scroll wheel` or `ctrl + right click`: zoom in/out
- `ctrl + middle click`: move the camera
- `left click` on an object: if the object has a mass and a collision shape, you can interact with it with the mouse
- `w`: wireframe (see collision shapes)
- `g`: show/hide menu
- `esc`: quit the simulator
"""

from itertools import count
import pyrobolearn as prl


# create simulator
sim = prl.simulators.Bullet(render=True)

# create basic world (i.e. with a floor and gravity enabled)
world = prl.worlds.BasicWorld(sim)

# load rrbot
robot = prl.robots.RRBot(sim)


# run simulator
for t in count():
    # perform a step in the simulator and sleep for `sim.dt` (which is 1./240 in this case)
    sim.step(sim.dt)
