#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Get the robot's joint position and velocity states.

In this example, we load a Kuka manipulator and print the joint position and velocity states. You can try to move the
robot and see how it affects the states.
"""

from itertools import count
import pyrobolearn as prl


# Create simulator
sim = prl.simulators.Bullet()

# create world
world = prl.worlds.BasicWorld(sim)

# load the kuka robot
robot = world.load_robot('kuka_iiwa')

# create joint states
state = prl.states.JointPositionState(robot, joint_ids=[1, 2]) + prl.states.JointVelocityState(robot, joint_ids=1)

# perform simulation
for t in count():

    # update the state
    state()  # or state.read()

    # print the state
    print(state)

    # step in simulation
    world.step(sleep_dt=sim.dt)
