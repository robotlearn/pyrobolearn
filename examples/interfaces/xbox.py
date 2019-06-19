#!/usr/bin/env python
"""Load the Xbox game controller interface
"""

import time
from itertools import count
import argparse

from pyrobolearn.tools.interfaces.controllers.xbox import Xbox360ControllerInterface, XboxOneControllerInterface

# create parser to select the game controller
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--controller', help='The Xbox game controller to use (xbox one or xbox 360)', type=str,
                    choices=['360', 'one'], default='one')
args = parser.parse_args()


# load corresponding Xbox controller interface
if args.controller == '360':
    controller = Xbox360ControllerInterface(verbose=True)
elif args.controller == 'one':
    controller = XboxOneControllerInterface(verbose=True)
else:
    raise NotImplementedError("Unknown game controller")


# run controller
print('Running controller...')
for _ in count():

    # run one step with the interface
    controller.run()  # same as `step()` if we are not using threads

    # get the last update and print it
    b = controller.last_updated_button
    print("Last updated button: {} with value: {}".format(b, controller[b]))

    # sleep a bit
    time.sleep(0.01)
