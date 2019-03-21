#!/usr/bin/env python
"""Define the various joint actions

This includes notably the joint positions, velocities, and force/torque actions.
"""

import numpy as np
from abc import ABCMeta
from robot_actions import RobotAction


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointAction(RobotAction):
    r"""Joint Action
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, joint_ids=None):
        """
        Initialize the joint action.

        Args:
            robot (Robot): robot instance
            joint_ids (int, int[N]): joint id or list of joint ids
        """
        super(JointAction, self).__init__(robot)

        # get the joints of the robot
        if joint_ids is None:
            joint_ids = robot.getJointIds()
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        self.joints = joint_ids

    # @property
    # def size(self):
    #     return len(self.joints)

    def bounds(self):
        return self.robot.getJointLimits(self.joints)


class JointPositionAction(JointAction):
    r"""Joint Position Action

    Set the joint positions using position control.
    """

    def __init__(self, robot, joint_ids=None, kp=None, kd=None, max_force=None):
        self.kp, self.kd, self.max_force = kp, kd, max_force
        super(JointPositionAction, self).__init__(robot, joint_ids)
        self.data = robot.getJointPositions(self.joints)

    def _write(self, data=None):
        if data is None:
            self.robot.setJointPositions(self._data, self.joints, kp=self.kp, kd=self.kd, maxTorque=self.max_force)
        else:
            self.robot.setJointPositions(data, self.joints, kp=self.kp, kd=self.kd, maxTorque=self.max_force)


class JointVelocityAction(JointAction):
    r"""Joint Velocity Action

    Set the joint velocities using velocity control.
    """

    def __init__(self, robot, joint_ids=None):
        super(JointVelocityAction, self).__init__(robot, joint_ids)
        self.data = robot.getJointVelocities(self.joints)

    def _write(self, data=None):
        if data is None:
            self.robot.setJointVelocities(self._data, self.joints)
        else:
            self.robot.setJointVelocities(data, self.joints)


class JointForceAction(JointAction):
    r"""Joint Force Action

    Set the joint force/torque using force/torque control.
    """

    def __init__(self, robot, joint_ids=None, f_min=-np.infty, f_max=np.infty):
        super(JointForceAction, self).__init__(robot, joint_ids)
        self.data = robot.getJointTorques(self.joints)
        self.f_min = f_min
        self.f_max = f_max

    def _write(self, data=None):
        if data is None:
            self.robot.setJointTorques(self._data, self.joints)
        else:
            data = np.clip(data, self.f_min, self.f_max)
            self.robot.setJointTorques(data, self.joints)


class JointAccelerationAction(JointAction):
    r"""Joint Acceleration Action

    Set the joint accelerations using force/torque control. In order to produce the given joint accelerations,
    we use inverse dynamics which given the joint accelerations produce the corresponding joint forces/torques
    to be applied.
    """

    def __init__(self, robot, joint_ids=None, a_min=-np.infty, a_max=np.infty):
        super(JointAccelerationAction, self).__init__(robot, joint_ids)
        self.data = robot.getJointAccelerations(self.joints)
        self.a_min = a_min
        self.a_max = a_max

    def _write(self, data=None):
        if data is None:
            self.robot.setJointAccelerations(self._data, self.joints)
        else:
            data = np.clip(data, self.a_min, self.a_max)
            self.robot.setJointAccelerations(data, self.joints)
