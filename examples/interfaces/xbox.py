#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load the Xbox game controller interface

How to run:
```
$ python xbox.py --help                         # for help
$ python xbox.py --controller xbox              # to use any Xbox game controller (by default)
$ python xbox.py --controller xbox-360          # to use Xbox 360 game controller
$ python xbox.py --controller xbox-one          # to use Xbox One game controller
```

Note that these game controllers are blocking by default, thus set `use_thread` to True to avoid to be blocked.
"""

import time
import argparse

from pyrobolearn.tools.interfaces.controllers.xbox import XboxControllerInterface, XboxOneControllerInterface, \
    Xbox360ControllerInterface


# create parser to select the game controller
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--use_thread', help='If we should run the Xbox controller in a thread.', type=bool,
                    default=True)
parser.add_argument('-c', '--controller', help='The Xbox game controller to use.', type=str,
                    choices=['xbox', 'xbox-360', 'xbox-one'], default='xbox')
args = parser.parse_args()


# load corresponding Xbox controller interface
if args.controller == 'xbox':
    controller = XboxControllerInterface(use_thread=args.use_thread, verbose=False)
elif args.controller == 'xbox-360':
    controller = Xbox360ControllerInterface(use_thread=args.use_thread, verbose=False)
elif args.controller == 'xbox-one':
    controller = XboxOneControllerInterface(use_thread=args.use_thread, verbose=False)
else:
    raise NotImplementedError("Unknown game controller")


# run controller
print('Running controller...')
while True:

    # run one step with the interface
    controller.step()

    # get the last update and print it
    b = controller.last_updated_button
    print("Last updated button: {} with value: {}".format(b, controller[b]))

    # sleep a bit
    time.sleep(0.01)
