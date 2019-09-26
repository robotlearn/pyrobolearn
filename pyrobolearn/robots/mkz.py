# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the Lincoln MKZ car robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.wheeled_robot import AckermannWheeledRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MKZ(AckermannWheeledRobot):
    r"""Lincoln MKZ car

    Drive-by-wire interface to the Dataspeed Inc. Lincoln MKZ DBW kit.

    References:
        - [1] Dataspeed Inc.: https://www.dataspeedinc.com/
        - [2] ROS wiki: http://wiki.ros.org/dbw_mkz
        - [3] Bitbucket: https://bitbucket.org/DataspeedInc/dbw_mkz_ros
    """

    def __init__(self, simulator, position=(0, 0, .4), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/mkz/mkz.urdf'):
        """
        Initialize the MKZ car.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.4)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.4,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(MKZ, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'mkz'

        self.wheels = [self.get_link_ids(link) for link in ['wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr']
                       if link in self.link_names]
        self.wheel_directions = np.ones(len(self.wheels))

        self.steering = [self.get_link_ids(link) for link in ['steer_fl', 'steer_fr']
                         if link in self.link_names]

    def steer(self, angle):
        """Set steering angle"""
        angle = angle * np.ones(len(self.steering))
        self.set_joint_positions(angle, joint_ids=self.steering)


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
    robot = MKZ(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        robot.drive_forward(2)
        world.step(sleep_dt=1./240)
