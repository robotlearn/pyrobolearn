#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Control a quadcopter in the air using speech.

Try to say:
- turn right/left
- move/go higher/lower/forward/backward/right/left
- go faster/slower
- anything else will ask the robot to hover
"""

from itertools import count

import pyrobolearn as prl
from pyrobolearn.tools.bridges.audio.robots.bridge_speech_quadcopter import BridgeSpeechRecognizerQuadcopter


# create simulator
sim = prl.simulators.Bullet()

# create basic world (with a floor and gravity enabled by default)
world = prl.worlds.BasicWorld(sim)

# load quadcopter
robot = prl.robots.Quadcopter(sim, position=[0., 0., 2.])
world.load_robot(robot)

# load bridge that connects the speech interface with the quadcopter
# Note that it will create automatically the Speech Recognizer interface inside the bridge
bridge = BridgeSpeechRecognizerQuadcopter(robot=robot, interface=None, verbose=True)


# run simulator
for t in count():
    # perform a step with the bridge and interface
    bridge.step(update_interface=True)

    # ask the world camera to follow the wheeled robot
    world.follow(robot, distance=2)

    # perform one step in the world
    world.step(sleep_dt=1. / 240)
