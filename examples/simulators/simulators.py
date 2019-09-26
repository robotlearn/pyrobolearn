#!/usr/bin/env python
"""Simulator tests.

Example on how to load different things with the simulators. This example is still in an experimental phase. For
now, only Bullet is fully-supported. We are working on the other ones, especially the Mujoco simulator.
- Bullet: OK
- Raisim: OK  (todo: for collision bodies, it only accepts OBJ files)
- MuJoCo: OK  (todo: control still missing)
- DART: OK, but capsules don't have collision shapes... (todo: fix some URDFs)
- VREP: Not implemented yet + problem when importing PyRep with pybullet. Also, need to figure out how to call the
  'loadURDF' plugin.
- Isaac: not available yet.
"""

import os
from itertools import count

from pyrobolearn.simulators.bullet import Bullet
from pyrobolearn.simulators.raisim import Raisim
from pyrobolearn.simulators.dart import Dart
from pyrobolearn.simulators.mujoco import Mujoco
# from pyrobolearn.simulators.vrep import VREP  # Problem when importing PyRep with Pybullet
# from pyrobolearn.simulators.isaac import Isaac  # Not available yet


sim = Bullet(render=True)
# sim = Raisim(render=True)
# sim = Dart(render=True)
# sim = Mujoco(render=True)
# sim = VREP(render=True)
# sim = Isaac(render=True)
print("Gravity: {}".format(sim.get_gravity()))

# load floor
floor = sim.load_floor(dimension=20)

# create box
box = sim.create_primitive_object(sim.GEOM_BOX, position=(0, 0, 2), mass=1, rgba_color=(1, 0, 0, 1))
sphere = sim.create_primitive_object(sim.GEOM_SPHERE, position=(2, 2, 2), mass=1, rgba_color=(0, 1, 0, 1))
cylinder = sim.create_primitive_object(sim.GEOM_CYLINDER, position=(0, 2, 2), mass=1)
capsule = sim.create_primitive_object(sim.GEOM_CAPSULE, position=(0, -2, 2), mass=1, rgba_color=(0, 0, 1, 1),
                                      radius=0.5, height=0.5)

# load robot
urdf_path = os.path.dirname(os.path.abspath(__file__)) + '/../../pyrobolearn/robots/urdfs/'
# path = urdf_path + 'rrbot/rrbot.urdf'
# path = urdf_path + 'jaco/jaco.urdf'
# path = urdf_path + 'kuka/kuka_iiwa/iiwa14.urdf'
# path = urdf_path + 'hyq2max/hyq2max.urdf'
path = urdf_path + 'anymal/anymal.urdf'
# path = urdf_path + 'centauro/centauro_stick.urdf'

robot = sim.load_urdf(path, position=(3, -3, 2), use_fixed_base=False)

# perform step
for t in count():
    sim.step(sleep_time=sim.dt)
