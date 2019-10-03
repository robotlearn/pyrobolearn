# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Get the cartesian position state of an object loaded in the world.

In this example, we load a sphere in the world which can be moved using the mouse. At every time step, we print the
cartesian position state of the sphere.
"""

from itertools import count
import pyrobolearn as prl


# Create simulator
sim = prl.simulators.Bullet()

# create world
world = prl.worlds.BasicWorld(sim)

# load a sphere in the world
sphere = world.load_sphere(position=(0., 0., 1.), mass=1., radius=0.2, color=(0.8, 0, 0, 1.), return_body=True)

# create state
state = prl.states.PositionState(sphere)

# perform simulation
for t in count():

    # update the state
    state()  # or state.read()

    # print the state
    print(state)

    # step in simulation
    world.step(sleep_dt=sim.dt)
