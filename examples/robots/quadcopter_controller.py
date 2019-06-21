#!/usr/bin/env python
"""Control a quadcopter in the air using an Xbox or Playstation game controller.

how to run:
```
$ python quadcopter_controller.py --help                    # for help
$ python quadcopter_controller.py --controller keyboard     # to use the keyboard
$ python quadcopter_controller.py --controller xbox         # to use Xbox game controller
$ python quadcopter_controller.py --controller ps          # to use PS game controller
```

Mapping of the keyboard interface:
- `top arrow`: move forward
- `bottom arrow`: move backward
- `left arrow`: move sideways to the left
- `right arrow`: move sideways to the right
- `ctrl + top arrow`: ascend
- `ctrl + bottom arrow`: descend
- `ctrl + left arrow`: turn to the right
- `ctrl + right arrow`: turn to the left
- `space`: switch between first-person and third-person view
"""

# import numpy as np
from itertools import count
import argparse

import pyrobolearn as prl


# create parser to select the game controller
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--controller', help='the controller to use', type=str,
                    choices=['keyboard', 'xbox', 'ps'], default='keyboard')
args = parser.parse_args()


# load corresponding interface
if args.controller == 'keyboard':  # keyboard interface
    from pyrobolearn.tools.interfaces.mouse_keyboard.mousekeyboard import MouseKeyboardInterface as Controller
    from pyrobolearn.tools.bridges.mouse_keyboard.bridge_mousekeyboard_quadcopter \
        import BridgeMouseKeyboardQuadcopter as Bridge
elif args.controller == 'xbox':  # Xbox interface
    from pyrobolearn.tools.interfaces.controllers.xbox import XboxControllerInterface as Controller
    from pyrobolearn.tools.bridges.controllers.robots.bridge_controller_quadcopter import BridgeControllerQuadcopter \
        as Bridge
elif args.controller == 'ps':  # PS interface
    from pyrobolearn.tools.interfaces.controllers.playstation import PSControllerInterface as Controller
    from pyrobolearn.tools.bridges.controllers.robots.bridge_controller_quadcopter import BridgeControllerQuadcopter \
        as Bridge
else:
    raise NotImplementedError("Unknown game controller")


# create simulator
sim = prl.simulators.Bullet()

# create basic world (with a floor and gravity enabled by default)
world = prl.worlds.BasicWorld(sim)

# load quadcopter
robot = prl.robots.Quadcopter(sim, position=[0., 0., 2.])
world.load_robot(robot)

# load interface that accepts input events
controller = Controller()

# load bridge that connects the interface/controller with the quadcopter
# The bridge is the one that maps the input events from the interface to commands sent to the quadcopter
bridge = Bridge(quadcopter=robot, interface=controller)

# run simulator
for t in count():
    # perform a step with the bridge and interface
    bridge.step(update_interface=True)

    # perform one step in the world
    world.step(sleep_dt=1. / 240)
