#!/usr/bin/env python
"""Provide the Husky robotic platform.
"""

import os
import numpy as np
from wheeled_robot import DifferentialWheeledRobot


class Husky(DifferentialWheeledRobot):
    r"""Husky robot

    Husky robot from Clearpath Robotics [1].

    References:
        [1] Clearpath Robotics: https://www.clearpathrobotics.com/husky-unmanned-ground-vehicle-robot/
        [2] ROS wiki: http://wiki.ros.org/Robots/Husky
        [3] Github: https://github.com/husky/husky
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, .14),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/husky/husky.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.14)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.14,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Husky, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'husky'

        self.wheels = [self.getLinkIds(link) for link in ['front_left_wheel_link', 'front_right_wheel_link',
                                                          'rear_left_wheel_link', 'rear_right_wheel_link']
                       if link in self.link_names]
        self.wheel_directions = np.ones(len(self.wheels))


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
    robot = Husky(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        robot.driveForward(2)
        world.step(sleep_dt=1./240)
