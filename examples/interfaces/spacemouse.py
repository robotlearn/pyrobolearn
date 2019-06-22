#!/usr/bin/env python
"""Run the Space mouse interface.

Make sure that the `spnav` library is installed, and the space mouse (by 3Dconnexion) is connected to the computer
before running this code.
"""

import argparse
import time

from pyrobolearn.tools.interfaces.mouse_keyboard.spacemouse import SpaceMouseInterface


# create parser
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--use_thread', help='If we should run the webcam interface in a thread.', type=bool,
                    default=True)
args = parser.parse_args()


# create webcam interface
if args.use_thread:
    # create and run interface in a thread
    interface = SpaceMouseInterface(use_thread=True, sleep_dt=0.001, verbose=True)
    raw_input('Press key to stop the space mouse interface')
else:
    # create interface
    interface = SpaceMouseInterface(use_thread=False, verbose=True)

    while True:
        # perform a `step` with the interface
        interface.step()
        time.sleep(0.001)
