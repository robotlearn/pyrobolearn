#!/usr/bin/env python
"""Provide the Laikago robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.legged_robot import QuadrupedRobot


class Laikago(QuadrupedRobot):
    r"""Laikago robot

    References:
        [1] Laikago: http://www.unitree.cc/e/action/ShowInfo.php?classid=6&id=1
        [2] https://github.com/erwincoumans/pybullet_robots/tree/master/data/laikago
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, .5),
                 init_orient=(0.5, 0.5, 0.5, 0.5),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/laikago/laikago.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.5)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.5,)
        if init_orient is None:
            init_orient = (0.5, 0.5, 0.5, 0.5)
        if useFixedBase is None:
            useFixedBase = False

        super(Laikago, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'laikago'

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['FL_hip_motor', 'FL_upper_leg', 'FL_lower_leg'],
                                   ['FR_hip_motor', 'FR_upper_leg', 'FR_lower_leg'],
                                   ['RL_hip_motor', 'RL_upper_leg', 'RL_lower_leg'],
                                   ['RR_hip_motor','RR_upper_leg', 'RR_lower_leg']]]

        self.feet = [self.getLinkIds(link) for link in ['FL_lower_leg', 'FR_lower_leg',
                                                        'RL_lower_leg', 'RR_lower_leg'] if link in self.link_names]

    def getHomeJointPositions(self):
        """Return the joint positions for the home position"""
        return np.zeros(self.getNumberOfDoFs())


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
    robot = Laikago(sim)

    # print information about the robot
    robot.printRobotInfo()

    # # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        robot.moveJointHomePositions()
        world.step(sleep_dt=1./240)
