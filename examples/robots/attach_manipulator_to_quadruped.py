#!/usr/bin/env python
"""Attach the Kuka Youbot manipulator to the HyQ2Max quadruped.
"""

import pyrobolearn as prl


# create simulator
sim = prl.simulators.Bullet()

# create world
world = prl.worlds.BasicWorld(sim)

# create quadruped and manipulator
quadruped = world.load_robot('hyq2max')
manipulator = world.load_robot('kuka_youbot_arm', position=(0., 0., 1.), fixed_base=False)

# attach manipulator to the back of the robot
world.attach(body1=quadruped, body2=manipulator, link1=-1, link2=-1, joint_axis=[0., 0., 0.],
             parent_frame_position=[0.25, 0., 0.2], child_frame_position=[0., 0., 0.])

# run simulation
for t in prl.count():
    sim.step(sim.dt)
