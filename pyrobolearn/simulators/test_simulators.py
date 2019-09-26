# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Simulator tests.

Tests with the simulators.
- Bullet: OK
- Raisim: OK
- MuJoCo: In progress...
- DART: OK, but capsule doesn't have a collision shape...
- VREP: Not done yet + problem when importing PyRep with pybullet. Also, I need to figure out how to call the
  'loadURDF' plugin.
"""

import os
import time
from itertools import count
from pyrobolearn.simulators.bullet import Bullet
from pyrobolearn.simulators.raisim import Raisim
from pyrobolearn.simulators.dart import Dart
from pyrobolearn.simulators.mujoco import Mujoco
# from pyrobolearn.simulators.vrep import VREP  # Problem when importing PyRep with Pybullet


sim = Bullet(render=True)
# sim = Raisim(render=True)
# sim = Dart(render=True)
# sim = MuJoCo(render=True)
# sim = VREP(render=True)
print("Gravity: {}".format(sim.get_gravity()))

# load floor
floor = sim.load_floor(dimension=20)

# create box
box = sim.create_primitive_object(sim.GEOM_BOX, position=(0, 0, 2), mass=1, rgba_color=(1, 0, 0, 1))
sphere = sim.create_primitive_object(sim.GEOM_SPHERE, position=(2, 2, 2), mass=1, rgba_color=(0, 1, 0, 1))
capsule = sim.create_primitive_object(sim.GEOM_CAPSULE, position=(0, -2, 2), mass=1, rgba_color=(0, 0, 1, 1))
cylinder = sim.create_primitive_object(sim.GEOM_CYLINDER, position=(0, 2, 2), mass=1)

# load robot
path = os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/anymal/anymal.urdf'
robot = sim.load_urdf(path, position=(3, -3, 2))

# perform step
for t in count():
    sim.step(sleep_time=sim.dt)
