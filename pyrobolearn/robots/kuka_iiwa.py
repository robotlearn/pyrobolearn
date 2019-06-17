#!/usr/bin/env python
"""Provide the Kuka IIWA robotic platform.
"""

import os

from pyrobolearn.robots.manipulator import ManipulatorRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class KukaIIWA(ManipulatorRobot):
    r"""Kuka IIWA robot

    IIWA stands for 'Intelligent Industrial Work Assistant'.  This robot has 7 DoFs, and an ATI F/T sensor at the
    end-effector. Payload of 14kg and a range of 820mm.

    References:
        [1] Kuka robotics: https://www.kuka.com/en-de
        [2] https://github.com/IFL-CAMP/iiwa_stack
        [3] https://github.com/bulletphysics/bullet3/tree/master/data/kuka_iiwa
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0),
                 orientation=(0, 0, 0, 1),
                 scale=1.,
                 fixed_base=True,
                 urdf=os.path.dirname(__file__) + '/urdfs/kuka/kuka_iiwa/iiwa14.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(KukaIIWA, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'kuka_iiwa'

        # self.disable_motor()


# Test
if __name__ == "__main__":
    import numpy as np
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = KukaIIWA(sim)

    # print information about the robot
    robot.print_info()
    # H = robot.get_mass_matrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    # print(robot.get_link_world_positions(flatten=False))

    K = 5000*np.identity(3)
    # D = 2 * np.sqrt(K)
    # D = np.zeros((3,3))
    D = 100 * np.identity(3)
    x_des = np.array([0.3, 0.0, 0.8])
    x_des = np.array([0.52557296, 0.09732758, 0.80817658])
    linkId = robot.get_link_ids('iiwa_link_ee')

    for i in count():
        # print(robot.get_link_world_positions(flatten=False))

        # get state
        q = robot.get_joint_positions()
        dq = robot.get_joint_velocities()
        x = robot.get_link_world_positions(linkId)
        dx = robot.get_link_world_linear_velocities(linkId)

        # get (linear) jacobian
        J = robot.get_linear_jacobian(linkId, q)

        # get coriolis, gravity compensation torques
        torques = robot.get_coriolis_and_gravity_compensation_torques(q, dq)

        # Impedance control: attractor point
        F = K.dot(x_des - x) - D.dot(dx)
        # F = -D.dot(dx)
        tau = J.T.dot(F)
        print("Torques: {}".format(tau))
        torques += tau
        robot.set_joint_torques(torques)

        # step in simulation
        world.step(sleep_dt=1./240)
