#!/usr/bin/env python
"""Provide the F10 racecar robotic platform.
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


class F10Racecar(AckermannWheeledRobot):
    r"""F10Racecar robot

    References:
        - [1] https://github.com/erwincoumans/pybullet_robots/tree/master/data/f10_racecar
    """

    def __init__(self, simulator, position=(0, 0, .1), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/f10_racecar/racecar.urdf'):  # racecar_differential.urdf
        """
        Initialize the F10 racecar.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[3]): Cartesian world position.
            orientation (np.array[4]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the car base will be fixed in the world.
            scale (float): scaling factor that is used to scale the car.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.1)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.1,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(F10Racecar, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'racecar'

        self.wheels = [self.get_link_ids(link) for link in ['left_front_wheel', 'right_front_wheel',
                                                            'left_rear_wheel', 'right_rear_wheel']
                       if link in self.link_names]
        self.wheel_directions = np.ones(len(self.wheels))

        self.steering = [self.get_link_ids(link) for link in ['left_steering_hinge', 'right_steering_hinge']
                         if link in self.link_names]

    def steer(self, angle):
        """Set steering angle"""
        angle = angle * np.ones(len(self.steering))
        self.set_joint_positions(angle, joint_ids=self.steering)


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = F10Racecar(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        robot.drive_forward(10)
        world.step(sleep_dt=1./240)
