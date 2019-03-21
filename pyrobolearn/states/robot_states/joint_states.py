#!/usr/bin/env python
"""Define the various joint states

This includes notably the joint positions, velocities, and force/torque states.
"""

from abc import ABCMeta

from pyrobolearn.states.robot_states import RobotState


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointState(RobotState):
    r"""Joint State of a robot (abstract class).
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, joint_ids=None):
        """
        Initialize the joint state.

        Args:
            robot (Robot): robot instance
            joint_ids (int, int[N]): joint id or list of joint ids
        """
        super(JointState, self).__init__(robot)

        # get the joints of the robot
        if joint_ids is None:
            joint_ids = robot.getJointIds()
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        self.joints = joint_ids

        # read the data
        self._read()


class JointPositionState(JointState):
    r"""Joint Position State

    Return the joint positions as the state.
    """

    def __init__(self, robot, joint_ids=None):
        super(JointPositionState, self).__init__(robot, joint_ids)

    def _read(self):
        self._data = self.robot.getJointPositions(self.joints)


class JointVelocityState(JointState):
    r"""Joint Velocity State

    Return the joint velocities as the state.
    """

    def __init__(self, robot, joint_ids=None):
        super(JointVelocityState, self).__init__(robot, joint_ids)

    def _read(self):
        self._data = self.robot.getJointVelocities(self.joints)


class JointForceTorqueState(JointState):
    r"""Joint Force Torque State

    Return the joint force and torques as the state.
    """

    def __init__(self, robot, joint_ids=None):
        super(JointForceTorqueState, self).__init__(robot, joint_ids)

    def _read(self):
        self._data = self.robot.getJointTorques(self.joints)


class JointAccelerationState(JointState):
    r"""Joint Acceleration State.

    Return the joint accelerations as the state. In order to produce the joint accelerations, we first read the
    joint torques and then applied forward dynamics to get the corresponding joint accelerations.
    """

    def __init__(self, robot, joint_ids=None):
        super(JointAccelerationState, self).__init__(robot, joint_ids)

    def _read(self):
        self._data = self.robot.getJointAccelerations(self.joints)
