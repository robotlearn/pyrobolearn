#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load the mouse keyboard interface.
"""

from itertools import count

import pyrobolearn as prl
from pyrobolearn.tools.interfaces.mouse_keyboard import MouseKeyboardInterface


# create simulator
sim = prl.simulators.Bullet()

# create mouse keyboard interface
# Note that to give the simulator `sim` to the interface can be optional especially if there is only one simulator.
# The interface will look in the memory to check the instantiated simulators and take the first one if it exists.
interface = MouseKeyboardInterface(sim)

# run interface
for _ in count():
    # perform a step with the interface
    interface.step()

    # print pressed keys
    if len(interface.key_pressed) > 0:
        print("Keys that are pressed: {}".format(interface.key_pressed))
    if len(interface.key_down) > 0:
        print("Keys that are down: {}".format(interface.key_down))

    # perform a step with the simulator
    sim.step(sleep_time=sim.dt)
