#!/usr/bin/env python
"""Provide the Soft Hand robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import SoftHand

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
left_hand = SoftHand(sim, init_pos=(-0.15, 0, 0), left=True)
right_hand = SoftHand(sim, init_pos=(0.15, 0., 0.), init_orient=(0, 0, 1, 0), left=False)

# print information about the robot
left_hand.printRobotInfo()
# H = left_hand.calculateMassMatrix()
# print("Inertia matrix: H(q) = {}".format(H))

# Position control using sliders
# left_hand.addJointSlider()

left_hand.setJointPositions([0.] * left_hand.getNumberOfDoFs())
right_hand.setJointPositions([0.] * right_hand.getNumberOfDoFs())

for i in count():
    # left_hand.updateJointSlider()
    # left_hand.setJointPositions([0.] * left_hand.getNumberOfDoFs())
    # right_hand.setJointPositions([0.] * right_hand.getNumberOfDoFs())

    world.step(sleep_dt=1./240)
