#!/usr/bin/env python
"""Load the Playstation game controller interface

How to run:
```
$ python playstation.py --help                   # for help
$ python playstation.py --controller ps          # to use any PS game controller (by default)
$ python playstation.py --controller ps3         # to use PS3 game controller
$ python playstation.py --controller ps4         # to use PS4 game controller
```
"""

import time
from itertools import count
import argparse

from pyrobolearn.tools.interfaces.controllers.playstation import *

# create parser to select the game controller
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--controller', help='The Playstation game controller to use (ps, ps3, or ps4)', type=str,
                    choices=['ps', 'ps3', 'ps4'], default='ps')
args = parser.parse_args()


# load corresponding Playstation controller interface
if args.controller == 'ps':
    controller = PSControllerInterface(verbose=False)
if args.controller == 'ps3':
    controller = PS3ControllerInterface(verbose=False)
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
