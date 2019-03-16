#!/usr/bin/env python
"""Provide the WAM robotic platform.
"""

import os
from manipulator import ManipulatorRobot


class WAM(ManipulatorRobot):
    r"""Wam robot

    References:
        [1] https://github.com/jhu-lcsr/barrett_model
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=True,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/wam/wam.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        super(WAM, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'wam'

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
    robot = WAM(sim)

    # print information about the robot
    robot.printRobotInfo()
    # H = robot.calculateMassMatrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    robot.setJointPositions([np.pi / 4, np.pi / 2], jointId=[0,1]) #2, 4])

    Jlin = robot.calculateJacobian(6, localPosition=(0., 0., 0.))[:3]
    robot.drawVelocityManipulabilityEllipsoid(6, Jlin, color=(1,0,0,0.7))
    for _ in range(5):
        world.step(sleep_dt=1./240)

    Jlin = robot.calculateJacobian(6, localPosition=(0., 0., 0.))[:3]
    robot.drawVelocityManipulabilityEllipsoid(6, Jlin, color=(0, 0, 1, 0.7))
    for _ in range(45):
        world.step(sleep_dt=1./240)

    Jlin = robot.calculateJacobian(6, localPosition=(0., 0., 0.))[:3]
    robot.drawVelocityManipulabilityEllipsoid(6, Jlin)

    for i in count():
        if i%1000 == 0:
            print("Joint Torques: {}".format(robot.getJointTorques()))
            print("Gravity Torques: {}".format(robot.getGravityCompensationTorques()))
            print("Compensation Torques: {}".format(robot.getCoriolisAndGravityCompensationTorques()))
        # step in simulation
        world.step(sleep_dt=1./240)
