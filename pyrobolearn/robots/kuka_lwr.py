#!/usr/bin/env python
"""Provide the Kuka LWR robotic platform.
"""

import os

from pyrobolearn.robots.manipulator import Manipulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class KukaLWR(Manipulator):
    r"""Kuka LWR robot

    LWR stands for 'Light Weight Robot'. This robot has 7 DoFs, and an ATI F/T sensor at the end-effector.
    Payload of 7kg and a range of 790mm.

    References:
        - [1] Kuka robotics: https://www.kuka.com/en-de
        - [2] https://github.com/CentroEPiaggio/kuka-lwr
        - [3] https://github.com/bulletphysics/bullet3/tree/master/data/kuka_lwr
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=True, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/kuka/kuka_lwr/kuka.urdf'):
        """
        Initialize the Kuka LWR robot.

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
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(KukaLWR, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'kuka_lwr'


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
    robot = KukaLWR(sim)

    # print information about the robot
    robot.print_info()
    # H = robot.get_mass_matrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
