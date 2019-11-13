#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the joint real-time plotting example.

Try to move the Kuka manipulator with your mouse, and check the joint values. Note that it can take few seconds to
load the plot.

Warnings: don't forget to close FIRST the figure, THEN the simulator otherwise you will have the plotting process still
running.
"""

from itertools import count
import pyrobolearn as prl


# create the simulator
sim = prl.simulators.Bullet()

# create the world
world = prl.worlds.BasicWorld(sim)

# load the robot
robot = world.load_robot('kuka_iiwa')

# create the joint real-time plotting tool
plot = prl.utils.plotting.JointRealTimePlot(robot, joint_ids=None, position=True, velocity=False,
                                            acceleration=False, torque=False, ticks=24)

# run the simulation
for t in count():
    # update the plot
    plot.update()

    # perform a step in the world
    world.step(sim.dt)
