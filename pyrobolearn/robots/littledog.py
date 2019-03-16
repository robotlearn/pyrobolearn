#!/usr/bin/env python
"""Provide the Little Dog robotic platform.
"""

import os
import numpy as np
from legged_robot import QuadrupedRobot


class LittleDog(QuadrupedRobot):
    r"""Little Dog

    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.2),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/littledog/littleDog.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.2)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.2,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(LittleDog, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'littledog'

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['front_left_hip', 'front_left_upper_leg', 'front_left_lower_leg'],
                                   ['front_right_hip', 'front_right_upper_leg', 'front_right_lower_leg'],
                                   ['back_left_hip', 'back_left_upper_leg', 'back_left_lower_leg'],
                                   ['back_right_hip', 'back_right_upper_leg', 'back_right_lower_leg']]]

        self.feet = [self.getLinkIds(link) for link in ['front_left_lower_leg', 'front_right_lower_leg',
                                                        'back_left_lower_leg', 'back_right_lower_leg']
                     if link in self.link_names]

        self.kp = 24. * np.ones(12)
        self.kd = np.array([0.5, 0.5, 0.16, 0.5, 0.5, 0.16, 0.5, 0.5, 0.16, 0.5, 0.5, 0.16])

        self.setJointPositions([-0.6, -0.6, 0.6, 0.6], self.feet)

        self.joint_nominal_config = np.array([0., 0., -0.6, 0., 0., -0.6, 0., 0., 0.6, 0., 0., 0.6])


# Tests
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = LittleDog(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        # robot.computeAndDrawCoMPosition()
        # robot.computeAndDrawProjectedCoMPosition()
        world.step(sleep_dt=1./240)
