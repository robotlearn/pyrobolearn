#!/usr/bin/env python
"""Provide the Kuka IIWA robotic platform.
"""

import os

from pyrobolearn.robots.manipulator import ManipulatorRobot


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
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 scaling=1.,
                 useFixedBase=True,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/kuka/kuka_iiwa/iiwa14.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        super(KukaIIWA, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'kuka_iiwa'

        # self.disableMotor()


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
    robot.printRobotInfo()
    # H = robot.calculateMassMatrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    # print(robot.getLinkWorldPositions(flatten=False))

    K = 5000*np.identity(3)
    # D = 2 * np.sqrt(K)
    # D = np.zeros((3,3))
    D = 100 * np.identity(3)
    x_des = np.array([0.3, 0.0, 0.8])
    x_des = np.array([0.52557296, 0.09732758, 0.80817658])
    linkId = robot.getLinkIds('iiwa_link_ee')

    for i in count():
        # print(robot.getLinkWorldPositions(flatten=False))

        # get state
        q = robot.getJointPositions()
        dq = robot.getJointVelocities()
        x = robot.getLinkWorldPositions(linkId)
        dx = robot.getLinkWorldLinearVelocities(linkId)

        # get (linear) jacobian
        J = robot.getLinearJacobian(linkId, q)

        # get coriolis, gravity compensation torques
        torques = robot.getCoriolisAndGravityCompensationTorques(q, dq)

        # Impedance control: attractor point
        F = K.dot(x_des - x) - D.dot(dx)
        # F = -D.dot(dx)
        tau = J.T.dot(F)
        print(tau)
        torques += tau
        robot.setJointTorques(torques)

        # step in simulation
        world.step(sleep_dt=1./240)
