#!/usr/bin/env python
"""Provide the universal robot platforms.

Specifically, it provides the classes for UR3, UR5, and UR10.

References:
    - [1] Universal robots: https://www.universal-robots.com/
    - [2] UR description: https://github.com/ros-industrial/universal_robot
"""

import os

from pyrobolearn.robots.manipulator import Manipulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class UR3(Manipulator):
    r"""UR3 manipulator

    References:
        - [1] Universal robots: https://www.universal-robots.com/
        - [2] UR description: https://github.com/ros-industrial/universal_robot
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=True, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/ur/ur3.urdf'):
        """
        Initialize the UR3 manipulator.

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
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(UR3, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'ur3'


class UR5(Manipulator):
    r"""UR5 manipulator

    References:
        - [1] Universal robots: https://www.universal-robots.com/
        - [2] UR description: https://github.com/ros-industrial/universal_robot
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=True, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/ur/ur5.urdf'):
        """
        Initialize the UR3 manipulator.

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
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(UR5, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'ur5'


class UR10(Manipulator):
    r"""UR10 manipulator

    References:
        - [1] Universal robots: https://www.universal-robots.com/
        - [2] UR description: https://github.com/ros-industrial/universal_robot
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=True, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/ur/ur10.urdf'):
        """
        Initialize the UR3 manipulator.

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
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(UR10, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'ur10'


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
    ur3 = UR3(sim, position=[0., -1., 0.])
    ur5 = UR5(sim, position=[0., 0., 0.])
    ur10 = UR10(sim, position=[0., 1., 0.])

    # print information about the robot
    ur3.print_info()
    # H = ur3.get_mass_matrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
