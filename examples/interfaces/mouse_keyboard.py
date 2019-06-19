#!/usr/bin/env python
"""Load the mouse keyboard interface.
"""

from itertools import count

import pyrobolearn as prl
from pyrobolearn.tools.interfaces.mouse_keyboard import MouseKeyboardInterface


# create simulator
sim = prl.simulators.Bullet()

# create mouse keyboard interface
interface = MouseKeyboardInterface(sim)

# run interface
for _ in count():
    # perform a step with the interface
    interface.step()

    # print pressed keys
    if len(interface.key_down) > 0:
        print("Keys that are pressed: {}".format(interface.key_down))

    # perform a step with the simulator
    sim.step(sleep_time=sim.dt)
