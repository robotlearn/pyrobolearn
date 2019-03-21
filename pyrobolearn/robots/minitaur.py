#!/usr/bin/env python
"""Provide the Minitaur robotic platform.
"""

import os
import collections
import numpy as np

from pyrobolearn.robots.legged_robot import QuadrupedRobot


class Minitaur(QuadrupedRobot):
    r"""Minitaur robot

    Minitaur robot from Ghost Robotics (https://www.ghostrobotics.io/)

    References:
        [1] pybullet_envs/bullet/minitaur.py
        [2] https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, .3),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 couple_legs=True,
                 foot_friction=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/minitaur/minitaur.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.3)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.3,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Minitaur, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'minitaur'

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['motor_front_leftL_link', 'lower_leg_front_leftL_link',
                                    'motor_front_leftR_link', 'lower_leg_front_leftR_link'],
                                   ['motor_front_rightL_link', 'lower_leg_front_rightL_link',
                                    'motor_front_rightR_link', 'lower_leg_front_rightR_link'],
                                   ['motor_back_leftL_link', 'lower_leg_back_leftL_link',
                                    'motor_back_leftR_link', 'lower_leg_back_leftR_link'],
                                   ['motor_back_rightL_link', 'lower_leg_back_rightL_link',
                                    'motor_back_rightR_link', 'lower_leg_back_rightR_link']]]

        self.feet = [[self.getLinkIds(link) for link in links if link in self.link_names]
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

        self.init_joint_positions = self.getHomeJointPositions()

        # constraints
        self.couple_legs = couple_legs
        if self.couple_legs:
            for leg in self.legs:
                self.sim.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=leg[3],
                                          childBodyUniqueId=self.id, childLinkIndex=leg[1],
                                          jointType=self.sim.JOINT_POINT2POINT,
                                          jointAxis=[0, 0, 0], parentFramePosition=[0, 0.005, 0.2],
                                          childFramePosition=[0, 0.01, 0.2])

        # disable motors
        self.disableMotor(self.knees)

        # kp, kd gains
        self.kp = 1.
        self.kd = 1.
        self.max_force = 3.5

        # set feet friction
        self.setFootFriction(friction=foot_friction, feet_id=self.feet)

        # set joint angles to home position
        self.setJointHomePositions()

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

    def getHomeJointPositions(self):
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

    def setJointPositions(self, position, jointId=None, kp=None, kd=None, velocity=None, maxTorque=None):
        if self.couple_legs:  # assume the joint ids are for the outer legs

            if jointId is None:
                pass

            # if the given joint is just one id
            elif isinstance(jointId, int):
                if jointId not in self.outer_joints:
                    raise ValueError("Expecting the jointId to be an outer joint as the legs of the minitaur are "
                                     "coupled")
                jointId = [jointId, jointId + 3]
                if isinstance(position, collections.Iterable):
                    position = position[0]

                position = np.array([position, -position])
                position += self.init_joint_positions[self.getQIndex(jointId)]

            # if multiple joint ids
            elif isinstance(jointId, collections.Iterable):
                # for each outer joint id, get the corresponding inner joint id
                joints = []
                for joint in jointId:
                    if joint not in self.outer_joints:
                        raise ValueError("One of the jointId is not an outer joint which is a problem as the legs of "
                                         "the minitaur are coupled")
                    joints.append(joint+3)

                # increase the list of joint ids to take into account inner joint ids
                jointId = list(jointId) + joints

                # compute the positions (offset original position, and compute positions for inner joints)
                position = list(position) + list(-position)
                position = np.array(position) + self.init_joint_positions[self.getQIndex(jointId)]

            else:
                raise TypeError("Unknown type of for jointId; expecting a list of int, or an int, got instead :"
                                "{}".format(type(jointId)))
        super(Minitaur, self).setJointPositions(position, jointId=jointId, kp=kp, kd=kd, velocity=velocity,
                                                maxTorque=maxTorque)


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
    robot = Minitaur(sim, couple_legs=True)

    # print information about the robot
    robot.printRobotInfo()
    print("Robot leg ids: {}".format(robot.legs))
    print("Robot feet ids: {}".format(robot.feet))

    # Position control using sliders
    # robot.addJointSlider(robot.getLeftFrontLegIds())

    t = 0
    # run simulator
    for _ in count():
        t += 0.01
        position = np.pi/4 * np.sin(2 * np.pi * t) * np.ones(len(robot.outer_hips))
        robot.setJointPositions(position, robot.outer_hips)
        # robot.updateJointSlider()
        # robot.computeAndDrawCoMPosition()
        # robot.computeAndDrawProjectedCoMPosition()
        world.step(sleep_dt=1./240)
