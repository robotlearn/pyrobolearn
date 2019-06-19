#!/usr/bin/env python
"""Control a quadcopter in the air using an Xbox or Playstation game controller.

how to run:
```
$ python quadcopter_controller.py --help                # for help
$ python quadcopter_controller.py --controller xbox     # to use Xbox game controller
$ python quadcopter_controller.py --controller ps3      # to use PS3 game controller
```
"""

import numpy as np
from itertools import count
import argparse

import pyrobolearn as prl


# create parser to select the game controller
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--controller', help='the game controller to use', type=str,
                    choices=['xbox', 'ps3'], default='ps3')
args = parser.parse_args()


# load corresponding interface
# if args.controller == 'xbox':
#     from pyrobolearn.tools.interfaces.controllers.xbox import Xbox360ControllerInterface as Controller
# elif args.controller == 'ps3':
#     from pyrobolearn.tools.interfaces.controllers.playstation import PS3ControllerInterface as Controller
# else:
#     raise NotImplementedError("Unknown game controller")


# create simulator
sim = prl.simulators.Bullet()

# create basic world (with a floor and gravity enabled by default)
world = prl.worlds.BasicWorld(sim)

# load quadcopter
robot = prl.robots.Quadcopter(sim, position=[0., 0., 2.])
world.load_robot(robot)

# load interface
# controller = Controller()

# run simulator
for t in count():
    # robot.hover()
    # robot.set_propeller_velocities(velocity)
    robot.move([1., 1., 1.])

    # follow quadcopter (seen from behind)
    world.follow(robot, distance=2, yaw=-np.pi / 2)

    # perform one step in the world
    world.step(sleep_dt=1. / 240)
