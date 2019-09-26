# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Control a wheeled robot using a keyboard, an Xbox or Playstation game controller.

how to run:
```
$ python wheeled_controller.py --help                    # for help
$ python wheeled_controller.py --controller keyboard     # to use the keyboard
$ python wheeled_controller.py --controller xbox         # to use Xbox game controller
$ python wheeled_controller.py --controller ps           # to use PS game controller
```

Mapping of the keyboard interface:
* `top arrow`: move forward
* `bottom arrow`: move backward
* `left arrow`: turn/steer to the left
* `right arrow`: turn/steer to the right

Mapping between the controller and the wheeled robot:
- left joystick: velocity of the wheeled robot
- east button (circle on PlayStation and B on Xbox): increase the speed
- west button (square on PlayStation and X on Xbox): decrease the speed
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
    from pyrobolearn.tools.bridges.mouse_keyboard.bridge_mousekeyboard_wheeled \
        import BridgeMouseKeyboardDifferentialWheeledRobot as Bridge
elif args.controller == 'xbox':  # Xbox interface
    from pyrobolearn.tools.interfaces.controllers.xbox import XboxControllerInterface as Controller
    from pyrobolearn.tools.bridges.controllers.robots.bridge_controller_wheeled import BridgeControllerWheeledRobot \
        as Bridge
elif args.controller == 'ps':  # PS interface
    from pyrobolearn.tools.interfaces.controllers.playstation import PSControllerInterface as Controller
    from pyrobolearn.tools.bridges.controllers.robots.bridge_controller_wheeled import \
        BridgeControllerWheeledRobot as Bridge
else:
    raise NotImplementedError("Unknown game controller")


# create simulator
sim = prl.simulators.Bullet()

# create basic world (with a floor and gravity enabled by default)
world = prl.worlds.BasicWorld(sim)

# load wheeled robot
robot = prl.robots.Epuck(sim, position=[0., 0.])
world.load_robot(robot)

# load interface that accepts input events
controller = Controller(use_thread=True, verbose=False)

# load bridge that connects the interface/controller with the quadcopter
# The bridge is the one that maps the input events from the interface to commands sent to the quadcopter
bridge = Bridge(robot=robot, interface=controller)


# run simulator
for t in count():
    # perform a step with the bridge and interface
    bridge.step(update_interface=True)

    # perform one step in the world
    world.step(sleep_dt=1. / 240)
