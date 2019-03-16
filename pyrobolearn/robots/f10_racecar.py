#!/usr/bin/env python
"""Provide the F10 racecar robotic platform.
"""

import os
import numpy as np
from wheeled_robot import AckermannWheeledRobot


class F10Racecar(AckermannWheeledRobot):
    r"""F10Racecar robot

    References:
        [1] https://github.com/erwincoumans/pybullet_robots/tree/master/data/f10_racecar
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, .1),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/f10_racecar/racecar.urdf'):  # racecar_differential.urdf
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.1)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.1,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(F10Racecar, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'racecar'

        self.wheels = [self.getLinkIds(link) for link in ['left_front_wheel', 'right_front_wheel',
                                                          'left_rear_wheel', 'right_rear_wheel']
                       if link in self.link_names]
        self.wheel_directions = np.ones(len(self.wheels))

        self.steering = [self.getLinkIds(link) for link in ['left_steering_hinge', 'right_steering_hinge']
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
    robot = F10Racecar(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        robot.driveForward(10)
        world.step(sleep_dt=1./240)
