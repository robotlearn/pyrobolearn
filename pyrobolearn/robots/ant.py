#!/usr/bin/env python
"""Provide the Ant Mujoco model.
"""

import os

from pyrobolearn.robots.legged_robot import QuadrupedRobot


class Ant(QuadrupedRobot):
    r"""Ant Mujoco Model
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.2),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scaling=1.,
                 urdf=os.path.dirname(__file__) + '/mjcfs/ant.xml'):
        # check parameters
        if position is None:
            position = (0., 0., 0.2)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.2,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Ant, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
        self.name = 'ant'

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['link0_3', 'link0_5'],
                                   ['link0_18', 'link0_20'],
                                   ['link0_13', 'link0_15'],
                                   ['link0_8', 'link0_10']]]

        self.feet = [self.get_link_ids(link) for link in ['front_left_foot', 'front_right_foot',
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
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider(robot.getLeftFrontLegIds() + robot.getRightFrontLegIds())

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        world.step(sleep_dt=1./240)
