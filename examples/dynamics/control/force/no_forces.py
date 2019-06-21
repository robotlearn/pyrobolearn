#!/usr/bin/env python
"""Force control: apply no forces/torques.

Try to move the end-effector using the mouse, and see what happens. Compare the obtained results with
`force/gravity_compensation.py` and `impedance/attractor_point.py`.
"""

import numpy as np
from itertools import count

import pyrobolearn as prl


# Create simulator
sim = prl.simulators.Bullet()

# create world
world = prl.worlds.BasicWorld(sim)

# load robot
robot = prl.robots.RRBot(sim)
robot.disable_motor()       # disable motors
robot.print_info()
robot.change_transparency()


# run simulator
for _ in count():

    # force control: apply no forces/torques
    robot.set_joint_torques(torques=np.zeros(len(robot.joints)))

    # perform a step in the world
    world.step(sleep_dt=sim.dt)
