#!/usr/bin/env python
"""Load the Playstation game controller interface
"""

import time
from itertools import count
import argparse

from pyrobolearn.tools.interfaces.controllers.playstation import PS3ControllerInterface, PS4ControllerInterface

# create parser to select the game controller
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--controller', help='The Playstation game controller to use (ps3 or ps4)', type=str,
                    choices=['ps3', 'ps4'], default='ps4')
args = parser.parse_args()


# load corresponding Playstation controller interface
if args.controller == 'ps3':
    controller = PS3ControllerInterface(verbose=True)
elif args.controller == 'ps4':
    controller = PS4ControllerInterface(verbose=False)
else:
    raise NotImplementedError("Unknown game controller")


# run controller
print('Running controller...')
for _ in count():

    # run one step with the interface
    controller.run()  # same as `step()` if we are not using threads

    # get the last update and print it
    b = controller.X
    print("X: {}".format(b))  # , controller[b]))

    # sleep a bit
    time.sleep(0.01)
