#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Minitaur robotic platform.
"""

import os
import collections.abc
import numpy as np

from pyrobolearn.robots.legged_robot import QuadrupedRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Minitaur(QuadrupedRobot):
    r"""Minitaur robot

    Minitaur robot from Ghost Robotics (https://www.ghostrobotics.io/)

    References:
        - [1] "Design Principles for a Family of Direct-Drive Legged Robots", Kenneally et al., 2016
        - [2] pybullet/gym/pybullet_envs/bullet/minitaur.py
        - [3] https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py
    """

    def __init__(self, simulator, position=(0, 0, .3), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 couple_legs=True, foot_friction=1., urdf=os.path.dirname(__file__) + '/urdfs/minitaur/minitaur.urdf'):
        """
        Initialize the Minitaur robot.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
            couple_legs (bool): if True, it will couple the legs by setting a constraint between two legs.
            foot_friction (float): foot friction value.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.3)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.3,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Minitaur, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'minitaur'

        self.base_height = 0.1638
        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['motor_front_leftL_link', 'lower_leg_front_leftL_link',
                                    'motor_front_leftR_link', 'lower_leg_front_leftR_link'],
                                   ['motor_front_rightL_link', 'lower_leg_front_rightL_link',
                                    'motor_front_rightR_link', 'lower_leg_front_rightR_link'],
                                   ['motor_back_leftL_link', 'lower_leg_back_leftL_link',
                                    'motor_back_leftR_link', 'lower_leg_back_leftR_link'],
                                   ['motor_back_rightL_link', 'lower_leg_back_rightL_link',
                                    'motor_back_rightR_link', 'lower_leg_back_rightR_link']]]

        self.feet = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['lower_leg_front_leftL_link', 'lower_leg_front_leftR_link'],
                                   ['lower_leg_front_rightL_link', 'lower_leg_front_rightR_link'],
                                   ['lower_leg_back_leftL_link', 'lower_leg_back_leftR_link'],
                                   ['lower_leg_back_rightL_link', 'lower_leg_back_rightR_link']]]

        self.outer_legs = [self.left_front_outer_leg, self.right_front_outer_leg,
                           self.left_back_outer_leg, self.right_back_outer_leg]
        self.inner_legs = [self.left_front_inner_leg, self.right_front_inner_leg,
                           self.left_back_inner_leg, self.right_back_inner_leg]

        self.outer_hips = [leg[0] for leg in self.outer_legs]
        self.inner_hips = [leg[0] for leg in self.inner_legs]
        self.outer_knees = [leg[1] for leg in self.outer_legs]
        self.inner_knees = [leg[1] for leg in self.inner_legs]

        self.hip_directions = [1, -1] * 4  # outer/inner
        self.knee_directions = [-1, 1] * 4  # outer/inner

        self.outer_joints = set([joint for leg in self.outer_legs for joint in leg])

        self.init_joint_positions = self.get_home_joint_positions()

        # constraints
        self.couple_legs = couple_legs
        if self.couple_legs:
            for leg in self.legs:
                self.sim.create_constraint(parent_body_id=self.id, parent_link_id=leg[3],
                                           child_body_id=self.id, child_link_id=leg[1],
                                           joint_type=self.sim.JOINT_POINT2POINT, joint_axis=[0, 0, 0],
                                           parent_frame_position=[0, 0.005, 0.2], child_frame_position=[0, 0.01, 0.2])

        # disable motors
        self.disable_motor(self.knees)

        # kp, kd gains
        self.kp = 1.
        self.kd = 1.
        self.max_force = 3.5

        # set feet friction
        self.set_foot_friction(frictions=foot_friction, feet_ids=self.feet)

        h = np.pi / 2  # hip angle from [2]
        k = 2.1834  # knee angle from [2]
        right_front_leg_initial_pos = [-h, k, -h, k]  # (outer, inner)
        right_back_leg_initial_pos = [h, -k, h, -k]  # (outer, inner)
        left_front_leg_initial_pos = [h, -k, h, -k]  # (outer, inner)
        left_back_leg_initial_pos = [-h, k, -h, k]  # (outer, inner)
        self._joint_configuration = {'home': np.array(right_front_leg_initial_pos + right_back_leg_initial_pos +
                                                      left_front_leg_initial_pos + left_back_leg_initial_pos),
                                     'standing': 'home',
                                     'init': 'home'}

        # set joint angles to home position
        self.set_home_joint_positions()

    ##############
    # Properties #
    ##############

    @property
    def hips(self):
        hip_ids = []
        for o, i in zip(self.outer_hips, self.inner_hips):
            hip_ids.append(o)
            hip_ids.append(i)
        return hip_ids

    @property
    def knees(self):
        knee_ids = []
        for o, i in zip(self.outer_knees, self.inner_knees):
            knee_ids.append(o)
            knee_ids.append(i)
        return knee_ids

    @property
    def left_front_outer_leg(self):
        return self.left_front_leg[:2]

    @property
    def left_front_inner_leg(self):
        return self.left_front_leg[2:]

    @property
    def right_front_outer_leg(self):
        return self.right_front_leg[2:]

    @property
    def right_front_inner_leg(self):
        return self.right_front_leg[:2]

    @property
    def left_back_outer_leg(self):
        return self.left_back_leg[:2]

    @property
    def left_back_inner_leg(self):
        return self.left_back_leg[2:]

    @property
    def right_back_outer_leg(self):
        return self.right_back_leg[2:]

    @property
    def right_back_inner_leg(self):
        return self.right_back_leg[:2]

    ###########
    # Methods #
    ###########

    def get_home_joint_positions(self):
        """Return the joint positions for the home position"""
        h = np.pi/2  # hip angle from [2]
        k = 2.1834  # knee angle from [2]
        # joint positions
        right_front_leg_initial_pos = [-h, k, -h, k]  # (outer, inner)
        right_back_leg_initial_pos = [h, -k, h, -k]  # (outer, inner)
        left_front_leg_initial_pos = [h, -k, h, -k]  # (outer, inner)
        left_back_leg_initial_pos = [-h, k, -h, k]  # (outer, inner)
        return np.array(right_front_leg_initial_pos + right_back_leg_initial_pos + left_front_leg_initial_pos +
                        left_back_leg_initial_pos)

    def set_joint_positions(self, positions, joint_ids=None, kp=None, kd=None, velocities=None, forces=None):
        if self.couple_legs:  # assume the joint ids are for the outer legs

            if joint_ids is None:
                pass

            # if the given joint is just one id
            elif isinstance(joint_ids, int):
                if joint_ids not in self.outer_joints:
                    raise ValueError("Expecting the jointId to be an outer joint as the legs of the minitaur are "
                                     "coupled")
                joint_ids = [joint_ids, joint_ids + 3]
                if isinstance(positions, collections.abc.Iterable):
                    positions = positions[0]

                positions = np.array([positions, -positions])
                positions += self.init_joint_positions[self.get_q_indices(joint_ids)]

            # if multiple joint ids
            elif isinstance(joint_ids, collections.abc.Iterable):
                # for each outer joint id, get the corresponding inner joint id
                joints = []
                for joint in joint_ids:
                    if joint not in self.outer_joints:
                        raise ValueError("One of the jointId is not an outer joint which is a problem as the legs of "
                                         "the minitaur are coupled")
                    joints.append(joint+3)

                # increase the list of joint ids to take into account inner joint ids
                joint_ids = list(joint_ids) + joints

                # compute the positions (offset original position, and compute positions for inner joints)
                positions = list(positions) + list(-positions)
                positions = np.array(positions) + self.init_joint_positions[self.get_q_indices(joint_ids)]

            else:
                raise TypeError("Unknown type of for jointId; expecting a list of int, or an int, got instead :"
                                "{}".format(type(joint_ids)))
        super(Minitaur, self).set_joint_positions(positions, joint_ids=joint_ids, kp=kp, kd=kd, velocities=velocities,
                                                  forces=forces)


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import Bullet
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = Minitaur(sim, couple_legs=True)

    # print information about the robot
    robot.print_info()
    # print("Robot leg ids: {}".format(robot.legs))
    # print("Robot feet ids: {}".format(robot.feet))

    # Position control using sliders
    # robot.add_joint_slider(robot.left_front_leg)

    t = 0
    # run simulator
    for _ in count():
        t += 0.01
        position = np.pi/4 * np.sin(2 * np.pi * t) * np.ones(len(robot.outer_hips))
        # print(robot.get_base_position())
        # robot.set_joint_positions(position, robot.outer_hips)
        # robot.update_joint_slider()
        # robot.compute_and_draw_com_position()
        # robot.compute_and_draw_projected_com_position()
        world.step(sleep_dt=1./240)
