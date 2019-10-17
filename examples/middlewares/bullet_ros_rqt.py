# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Test the publisher using PyBullet, ROS, and RQT.
"""

import pyrobolearn as prl

# create middleware and simulator
ros = prl.middlewares.ROS(publish=True, teleoperate=True)
sim = prl.simulators.Bullet(middleware=ros)

# launch RQT
ros.launch_gui()

# load world
world = prl.worlds.BasicWorld(sim)

# load robot
robot = world.load_robot('rrbot')

# run simulation
for t in prl.count():
    # get the joint positions from the Bullet simulator (because :attr:`teleoperate` has been set to True,
    # it will publish these read positions on the corresponding topic)
    q = robot.get_joint_positions()

    # perform a step in the simulator (and sleep for `sim.dt`)
    world.step(sim.dt)
