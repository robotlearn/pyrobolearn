#!/usr/bin/env python
"""Provide the Franka Emika robotic platform.
"""

import os

from pyrobolearn.robots.manipulator import Manipulator
from pyrobolearn.robots.gripper import ParallelGripper


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Franka(Manipulator):
    r"""Franka Emika Panda robot

    Warnings: CURRENTLY, THE INERTIAL TAGS ARE NOT CORRECT IN THE URDF!! I INVENTED THEM MYSELF BASED ON THE VOLUME,
    UNIFORM DENSITY, AND SUPPOSED MASS.

    References:
        - [1] Documentation: https://frankaemika.github.io/docs/index.html
        - [2] Overview: https://frankaemika.github.io/docs/overview.html
        - [3] C++ library: https://github.com/frankaemika/libfranka
        - [4] ROS integration: https://github.com/frankaemika/franka_ros
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), scale=1., fixed_base=True,
                 urdf=os.path.dirname(__file__) + '/urdfs/franka/franka.urdf'):
        """
        Initialize the Franka Emika Panda manipulator.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[3]): Cartesian world position.
            orientation (np.array[4]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.0,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(Franka, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'franka'

        # self.disable_motor()


class FrankaGripper(ParallelGripper):
    r"""Franka Emika Panda gripper

    Warnings: CURRENTLY, THE INERTIAL TAGS ARE NOT CORRECT IN THE URDF!! I INVENTED THEM MYSELF BASED ON THE VOLUME,
    UNIFORM DENSITY, AND SUPPOSED MASS.

    References:
        - [1] Documentation: https://frankaemika.github.io/docs/index.html
        - [2] Overview: https://frankaemika.github.io/docs/overview.html
        - [3] C++ library: https://github.com/frankaemika/libfranka
        - [4] ROS integration: https://github.com/frankaemika/franka_ros
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0., 0, 1.), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/franka/franka_gripper.urdf'):
        """
        Initialize the Franka Emika Panda gripper.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[3]): Cartesian world position.
            orientation (np.array[4]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
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
            orientation = (0, 0, 0, 1.)
        if fixed_base is None:
            fixed_base = True

        super(FrankaGripper, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'franka_gripper'


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
    robot = Franka(sim)

    # print information about the robot
    robot.print_info()
    # H = robot.get_mass_matrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    # Position control using sliders
    # robot.add_joint_slider()

    for i in count():
        # robot.update_joint_slider()
        # step in simulation
        world.step(sleep_dt=1./240)
