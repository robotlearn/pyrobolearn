# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the WAM robotic platform and the Barrett hand gripper.
"""

import os

from pyrobolearn.robots.manipulator import Manipulator
from pyrobolearn.robots.gripper import AngularGripper

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WAM(Manipulator):
    r"""Wam robot

    References:
        - [1] https://advanced.barrett.com/wam-arm-1
        - [2] https://github.com/jhu-lcsr/barrett_model
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=True, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/wam/wam.urdf'):
        """
        Initialize the WAM robot.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the gripper.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(WAM, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'wam'

        # self.disable_motor()


class BarrettHand(AngularGripper):
    r"""BarretHand (Gripper)

    References:
        - [1] https://advanced.barrett.com/wam-arm-1
        - [2] https://github.com/jhu-lcsr/barrett_model
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/wam/wam_gripper.urdf'):
        """
        Initialize the Barrett hand/gripper.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the gripper will be fixed in the world.
            scale (float): scaling factor that is used to scale the gripper.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(BarrettHand, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'barrett_hand'


# alias
WAMGripper = BarrettHand


# Test
if __name__ == "__main__":
    import numpy as np
    from itertools import count
    from pyrobolearn.simulators import Bullet
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = WAM(sim)

    # print information about the robot
    robot.print_info()
    # H = robot.get_mass_matrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    robot.set_joint_positions([np.pi / 4, np.pi / 2], joint_ids=[0, 1])  # 2, 4])

    Jlin = robot.get_jacobian(6)[:3]
    robot.draw_velocity_manipulability_ellipsoid(6, Jlin, color=(1, 0, 0, 0.7))
    for _ in range(5):
        world.step(sleep_dt=1./240)

    Jlin = robot.get_jacobian(6)[:3]
    robot.draw_velocity_manipulability_ellipsoid(6, Jlin, color=(0, 0, 1, 0.7))
    for _ in range(45):
        world.step(sleep_dt=1./240)

    Jlin = robot.get_jacobian(6)[:3]
    robot.draw_velocity_manipulability_ellipsoid(6, Jlin)

    for i in count():
        if i%1000 == 0:
            print("Joint Torques: {}".format(robot.get_joint_torques()))
            print("Gravity Torques: {}".format(robot.get_gravity_compensation_torques()))
            print("Compensation Torques: {}".format(robot.get_coriolis_and_gravity_compensation_torques()))
        # step in simulation
        world.step(sleep_dt=1./240)
