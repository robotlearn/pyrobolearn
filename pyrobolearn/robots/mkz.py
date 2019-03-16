#!/usr/bin/env python
"""Provide the Lincoln MKZ car robotic platform.
"""

import os
import numpy as np
from wheeled_robot import AckermannWheeledRobot


class MKZ(AckermannWheeledRobot):
    r"""Lincoln MKZ car

    Drive-by-wire interface to the Dataspeed Inc. Lincoln MKZ DBW kit.

    References:
        [1] Dataspeed Inc.: https://www.dataspeedinc.com/
        [2] ROS wiki: http://wiki.ros.org/dbw_mkz
        [3] Bitbucket: https://bitbucket.org/DataspeedInc/dbw_mkz_ros
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, .4),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/mkz/mkz.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.4)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.4,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(MKZ, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'mkz'

        self.wheels = [self.getLinkIds(link) for link in ['wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr']
                       if link in self.link_names]
        self.wheel_directions = np.ones(len(self.wheels))

        self.steering = [self.getLinkIds(link) for link in ['steer_fl', 'steer_fr']
                         if link in self.link_names]

    def setSteering(self, angle):
        """Set steering angle"""
        angle = angle * np.ones(len(self.steering))
        self.setJointPositions(angle, jointId=self.steering)


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
    robot = MKZ(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        robot.driveForward(2)
        world.step(sleep_dt=1./240)
