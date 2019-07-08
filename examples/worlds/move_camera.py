#!/usr/bin/env python
"""Move main camera in the world.

Move the main camera in the world using the keyboard:
- top arrow: move forward
- bottom arrow: move backward
- left arrow: move sideways to the left
- right arrow: move sideways to the right
- ctrl + top arrow: turn downward
- ctrl + bottom arrow: turn upward
- ctrl + left arrow: turn to the right
- ctrl + right arrow: turn to the left
"""

from itertools import count

import pyrobolearn as prl

# create simulator
sim = prl.simulators.Bullet()

# create basic world (with a floor and gravity enabled by default)
world = prl.worlds.BasicWorld(sim)

# get world camera
camera = world.camera

# create mouse-keyboard interface
interface = prl.tools.interfaces.MouseKeyboardInterface(sim)

# run simulator
for _ in count():
    # perform a step with the interface (i.e. get events from interface)
    interface.step()

    # check the keys that are down
    key, key_down = interface.key, interface.key_down
    if key.ctrl in key_down:
        if key.top_arrow in key_down:
            camera.pitch -= 0.005                                   # turn downward
        elif key.bottom_arrow in key_down:
            camera.pitch += 0.005                                   # turn upward
        elif key.left_arrow in key_down:
            camera.yaw -= 0.005                                     # turn to the right
        elif key.right_arrow in key_down:
            camera.yaw += 0.005                                     # turn to the left
    else:
        if key.top_arrow in key_down:
            camera.target_position += 0.01 * camera.forward_vector  # move forward
        elif key.bottom_arrow in key_down:
            camera.target_position -= 0.01 * camera.forward_vector  # move backward
        elif key.left_arrow in key_down:
            camera.target_position -= 0.01 * camera.lateral_vector  # move to the left
        elif key.right_arrow in key_down:
            camera.target_position += 0.01 * camera.lateral_vector  # move to the right

    # perform one step in the world
    world.step(sleep_dt=1. / 254)
