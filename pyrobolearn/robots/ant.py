#!/usr/bin/env python
"""Provide the Ant Mujoco model.
"""

import os
from legged_robot import QuadrupedRobot


class Ant(QuadrupedRobot):
    r"""Ant Mujoco Model
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.2),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/mjcfs/ant.xml'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.2)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.2,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Ant, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'ant'

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['link0_3', 'link0_5'],
                                   ['link0_18', 'link0_20'],
                                   ['link0_13', 'link0_15'],
                                   ['link0_8', 'link0_10']]]

        self.feet = [self.getLinkIds(link) for link in ['front_left_foot', 'front_right_foot',
                                                        'left_back_foot', 'right_back_foot']
                     if link in self.link_names]


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
    robot = Ant(sim)  # , useFixedBase=True)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider(robot.getLeftFrontLegIds() + robot.getRightFrontLegIds())

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        world.step(sleep_dt=1./240)
