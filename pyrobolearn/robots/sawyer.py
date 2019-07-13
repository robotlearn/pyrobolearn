#!/usr/bin/env python
"""Provide the Sawyer robotic platform.
"""

import os

from pyrobolearn.robots.wheeled_robot import WheeledRobot
from pyrobolearn.robots.manipulator import Manipulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Sawyer(Manipulator, WheeledRobot):
    r"""Sawyer robot

    Sawyer robot built by Rethink Robotics.

    References:
        - [1] Rethink Robotics
        - [2] https://github.com/RethinkRobotics/sawyer_robot
    """

    def __init__(self, simulator, position=(0, 0, 0.92), orientation=(0, 0, 0, 1), fixed_base=True, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/sawyer/sawyer.urdf'):
        """
        Initialize the Sawyer robot.

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
            position = (0., 0., 0.92)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.92,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(Sawyer, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'sawyer'

        self.head = self.get_link_ids('head') if 'head' in self.link_names else None

        self.arms = [[self.get_link_ids(link) for link in ['right_l0', 'right_l1', 'right_l2', 'right_l3', 'right_l4',
                                                         'right_l5', 'right_l6'] if link in self.link_names]]

        self.hands = [self.get_link_ids(link) for link in ['right_l6'] if link in self.link_names]


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
    robot = Sawyer(sim)

    # print information about the robot
    robot.print_info()

    # # Position control using sliders
    robot.add_joint_slider()

    # run simulator
    for _ in count():
        robot.update_joint_slider()
        world.step(sleep_dt=1./240)
