#!/usr/bin/env python
"""Provide the Franka Emika robotic platform.
"""

import os
from manipulator import ManipulatorRobot


class Franka(ManipulatorRobot):
    r"""Franka Emika robot

    WARNING: CURRENTLY, THE INERTIAL TAGS ARE NOT SET IN THE URDF!!

    References:
        [1] Documentation: https://frankaemika.github.io/docs/index.html
        [2] Overview: https://frankaemika.github.io/docs/overview.html
        [3] C++ library: https://github.com/frankaemika/libfranka
        [2] ROS integration: https://github.com/frankaemika/franka_ros
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 scaling=1.,
                 useFixedBase=True,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/franka/franka.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.0,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        super(Franka, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'franka'

        # self.disableMotor()


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
    robot.printRobotInfo()
    # H = robot.calculateMassMatrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    # Position control using sliders
    # robot.addJointSlider()

    for i in count():
        # robot.updateJointSlider()
        # step in simulation
        world.step(sleep_dt=1./240)
