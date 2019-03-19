#!/usr/bin/env python
"""Load the Allegro hand.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import AllegroHand

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
right_hand = AllegroHand(sim)  # , init_pos=(0.,0.,0.), init_orient=(0,0,1,0))

# print information about the robot
right_hand.printRobotInfo()
# H = right_hand.calculateMassMatrix()
# print("Inertia matrix: H(q) = {}".format(H))

# Position control using sliders
right_hand.addJointSlider()

for i in count():
    right_hand.updateJointSlider()
    # right_hand.setJointPositions([0.] * right_hand.getNumberOfDoFs())

    # step in simulation
    world.step(sleep_dt=1./240)
