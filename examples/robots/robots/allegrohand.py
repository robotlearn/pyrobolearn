# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Load the Allegro hand.
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import AllegroHand

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
right_hand = AllegroHand(sim)  # , position=(0.,0.,0.), orientation=(0,0,1,0))

# print information about the robot
right_hand.print_info()
# H = right_hand.get_mass_matrix()
# print("Inertia matrix: H(q) = {}".format(H))

# Position control using sliders
right_hand.add_joint_slider()

for i in count():
    right_hand.update_joint_slider()
    # right_hand.set_joint_positions([0.] * right_hand.num_dofs)

    # step in simulation
    world.step(sleep_dt=1./240)
