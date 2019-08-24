#!/usr/bin/env python
"""Force control: gravity compensation with RRBot

Try to move the end-effector using the mouse, and see what happens. Compare the obtained results with
`force/no_forces.py` and `impedance/attractor_point.py`.
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
robot.disable_motor()       # disable motors; comment the `robot.set_joint_torques(torques)` to see what happens
robot.print_info()
robot.change_transparency()
world.load_robot(robot)


# run simulator
for _ in count():
    # get current joint positions, velocities, accelerations
    q = robot.get_joint_positions()
    dq = robot.get_joint_velocities()
    ddq = np.zeros(len(q))

    # compute torques (Coriolis, centrifugal and gravity compensation) using inverse dynamics
    torques = robot.calculate_inverse_dynamics(ddq, dq, q)

    # force control
    robot.set_joint_torques(torques=torques)

    # perform a step in the world
    world.step(sleep_dt=sim.dt)
