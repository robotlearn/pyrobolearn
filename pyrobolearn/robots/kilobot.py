#!/usr/bin/env python
"""Provide the Kilobot robotic platform.
"""

# TODO: finish URDF: fix mass, inertia, dimensions, linear joint (spring mass)
# TODO: implement LRA vibration motor

import os
import numpy as np

from pyrobolearn.robots.robot import Robot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Kilobot(Robot):
    r"""Kilobot robot

    The Kilobot robot [1,2,3,4] is a small robot (diameter=33mm, height=34mm) mostly used in swarm robotics.
    It notably uses 2 coin shaped vibration motors [5] allowing the robot to move in a differential drive manner using
    the slip-stick principle.

    There are two types of vibration motors:
    - eccentric rotating mass vibration motor (ERM) [5.1]
    - linear resonant actuator (LRA) [5.2]

    References:
        [1] "Kilobot: a Low Cost Scalable Robot System for Collective Behaviors", Rubenstein et al., 2012
        [2] "Programmable self-assembly in a thousand-robot swarm", Rubenstein et al., 2014
        [3] Harvard's Self-Organizing Systems Research Group: https://ssr.seas.harvard.edu/kilobots
        [4] K-Team Corporation: https://www.k-team.com/mobile-robotics-products/kilobot
        [5] Precision Micro drives: https://www.precisionmicrodrives.com/
            - ERM: https://www.precisionmicrodrives.com/vibration-motors/
            - LRA: https://www.precisionmicrodrives.com/vibration-motors/linear-resonant-actuators-lras/
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/kilobot/kilobot.urdf'):  # TODO: finish URDF
        # check parameters
        if position is None:
            position = (0., 0., 0)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.0,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Kilobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'kilobot'

        # 2 coin shaped vibration motors with 255 different power levels
        self.motors = []

    def drive(self, values):
        """
        Drive the kilobot in a differential drive manner using the slip-stick principle.

        Args:
            values (float, int, np.array): 255 different power levels [0,255] for each motor.
        """
        if isinstance(values, (int, float)):
            values = np.ones(len(self.motors)) * values
        pass


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import Bullet
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # create robot
    robots = []
    for _ in range(30):
        x, y = np.random.uniform(low=-1, high=1, size=2)
        robot = world.load_robot(Kilobot, position=(x, y, 0))
        robots.append(robot)

    # print information about the robot
    robots[0].print_info()

    # Position control using sliders
    # robots[0].add_joint_slider()

    # run simulator
    for _ in count():
        # robots[0].update_joint_slider()
        for robot in robots:
            robot.drive(5)
        world.step(sleep_dt=1./240)
