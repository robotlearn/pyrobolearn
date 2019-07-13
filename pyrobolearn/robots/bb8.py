#!/usr/bin/env python
"""Provide the BB8 robotic platform.
"""

import os

from pyrobolearn.robots.robot import Robot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BB8(Robot):
    r"""BB8 robot

    References:
        - [1] https://github.com/eborghi10/BB-8-ROS
        - [2] http://www.theconstructsim.com/bb-8-gazebo-model/
    """

    def __init__(self, simulator, position=(0, 0, 0.4), orientation=(0, 0, 0, 1), fixed_base=False,
                 scale=1., urdf=os.path.dirname(__file__) + '/urdfs/bb8/bb8.urdf'):
        """
        Initialize the BB8 robot.

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
            position = (0., 0., 0.4)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.4,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(BB8, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'bb8'


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
    robot = BB8(sim)

    # print information about the robot
    robot.print_info()

    for i in count():
        robot.set_joint_velocities([0, -1, 0])

        # step in simulation
        world.step(sleep_dt=1./240)
