# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Load the Quadcopter robotic platform.
"""

import numpy as np
from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Quadcopter
from pyrobolearn.utils.units import rpm_to_rad_per_second


# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = Quadcopter(sim)

# print information about the robot
robot.print_info()

rpm = robot.get_stationary_rpm()
print("Stationary RPM: {}".format(rpm))
v = rpm_to_rad_per_second(rpm + 20)
v = [v, -v, v, -v]

# run simulation
for i in count():
    robot.set_joint_velocities(v)
    # step in simulation
    world.step(sleep_dt=1./240)
