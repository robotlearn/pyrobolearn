# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the Soft Hand robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import SoftHand

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
left_hand = SoftHand(sim, position=(-0.15, 0, 0), left=True)
right_hand = SoftHand(sim, position=(0.15, 0., 0.), orientation=(0, 0, 1, 0), left=False)

# print information about the robot
left_hand.print_info()
# H = left_hand.get_mass_matrix()
# print("Inertia matrix: H(q) = {}".format(H))

# Position control using sliders
# left_hand.add_joint_slider()

left_hand.set_joint_positions([0.] * left_hand.num_dofs)
right_hand.set_joint_positions([0.] * right_hand.num_dofs)

for i in count():
    # left_hand.update_joint_slider()
    # left_hand.set_joint_positions([0.] * left_hand.getNumberOfDoFs())
    # right_hand.set_joint_positions([0.] * right_hand.getNumberOfDoFs())

    world.step(sleep_dt=1./240)
