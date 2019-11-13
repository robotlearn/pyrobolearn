#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the link frame real-time plotting example.

Try to move the Kuka manipulator and / or the box with your mouse. Note that it can take few seconds to load the plot.

Warnings: don't forget to close FIRST the figure, THEN the simulator otherwise you will have the plotting process still
running.
"""

import pyrobolearn as prl


# create the simulator
sim = prl.simulators.Bullet()

# create the world
world = prl.worlds.BasicWorld(sim)

# load the robot and a box
robot = world.load_robot('kuka_iiwa')
box = world.load_box([0.7, 0., 0.2], dimensions=(0.2, 0.2, 0.2), color=(0.2, 0.2, 0.8, 1.), return_body=True)

# create the link frame real-time plotting tool
plot = prl.utils.plotting.LinkFrameRealTimePlot(bodies=[robot, box], link_ids=None, ticks=24)

# run the simulation
for t in prl.count():
    # update the plot
    plot.update()

    # perform a step in the world
    world.step(sim.dt)
