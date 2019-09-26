# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Load the Playstation game controller interface

How to run:
```
$ python playstation.py --help                   # for help
$ python playstation.py --controller ps          # to use any PS game controller (by default)
$ python playstation.py --controller ps3         # to use PS3 game controller
$ python playstation.py --controller ps4         # to use PS4 game controller
```

Note that these game controllers are blocking by default, thus set `use_thread` to True to avoid to be blocked.
"""

import time
import argparse

from pyrobolearn.tools.interfaces.controllers.playstation import *


# create parser to select the game controller
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--use_thread', help='If we should run the PlayStation controller in a thread.', type=bool,
                    default=True)
parser.add_argument('-c', '--controller', help='The Playstation game controller to use (ps, ps3, or ps4)', type=str,
                    choices=['ps', 'ps3', 'ps4'], default='ps')
args = parser.parse_args()


# load corresponding Playstation controller interface
if args.controller == 'ps':
    controller = PSControllerInterface(use_thread=args.use_thread, verbose=False)
if args.controller == 'ps3':
    controller = PS3ControllerInterface(use_thread=args.use_thread, verbose=False)
elif args.controller == 'ps4':
    controller = PS4ControllerInterface(use_thread=args.use_thread, verbose=False)
else:
    raise NotImplementedError("Unknown game controller")


# run controller
print('Running controller...')
while True:

    # run one step with the interface
    controller.step()  # same as `step()` if we are not using threads

    # get the last update and print it
    b = controller.last_updated_button
    print("Last updated button: {} with value: {}".format(b, controller[b]))

    # sleep a bit
    time.sleep(0.01)
