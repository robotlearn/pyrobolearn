# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define some joint terminal conditions for the environment.
"""

import copy
import numpy as np
from abc import ABCMeta

from pyrobolearn.terminal_conditions.robot_condition import RobotCondition


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointCondition(RobotCondition):
    r"""Joint Terminal Condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the joint states are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the joint states is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)

    States include positions, velocities, accelerations and torques.
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, joint_ids=None, bounds=(None, None), out=False, stay=False, all=False):
        """
        Initialize the joint terminal condition.

        Args:
            robot (Robot): robot instance
            joint_ids (int, int[N], None): joint id or list of joint ids
            bounds (tuple of float / np.array[N]): bounds to stay in/out or reach/leave.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the joint state
                leaves the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the joint state leaves the bounds, it results in a success.
            all (bool): this is only used if they are multiple joints. if True, all the joints are checked such that
                they are inside or outside the bounds depending on the other parameters. if False, any joints will be
                checked.
        """
        super(JointCondition, self).__init__(robot, bounds=bounds, dim=None, out=out, stay=stay, all=all)

        # get the joints of the robot
        if joint_ids is None:
            joint_ids = robot.get_joint_ids()
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        self.joints = joint_ids

        # check the bounds
        self.bounds = self._check_bounds(bounds=bounds)

    def _check_bounds(self, bounds):
        """Check the given bounds."""
        # check the type of the bounds
        if not isinstance(bounds, (tuple, list, np.ndarray)):
            raise TypeError("Expecting the given bounds to be a tuple/list/np.ndarray of float, instead got: "
                            "{}".format(type(bounds)))

        # check that the bounds have a length of 2 (i.e. lower and upper bounds)
        if len(bounds) != 2:
            raise ValueError("Expecting the bounds to be of length 2 (i.e. lower and upper bounds), instead got a "
                             "length of {}".format(len(bounds)))

        # if one of the bounds is None, raise error
        if bounds[0] is None or bounds[1] is None:
            raise ValueError("Expecting the bounds to not have None, but got: {}".format(bounds))

        # reshape bounds if necessary
        bounds = np.asarray(bounds).reshape(2, -1)
        if len(self.joints) != bounds.shape[1]:
            if bounds.shape[1] == 1:
                bounds = np.array([bounds[0, 0] * np.ones(len(self.joints)),
                                   bounds[1, 0] * np.ones(len(self.joints))])
            else:
                raise ValueError("Expecting the number of bounds (={}) to match up with the number of joints "
                                 "(={})".format(bounds.shape[1], len(self.joints)))
        return bounds


class JointPositionCondition(JointCondition):
    r"""Joint position terminal condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the joint positions are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the joint positions is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)
    """

    def __init__(self, robot, joint_ids=None, bounds=(None, None), out=False, stay=False, all=False):
        """
        Initialize the joint position terminal condition.

        Args:
            robot (Robot): robot instance
            joint_ids (int, int[N], None): joint id or list of joint ids
            bounds (tuple of float / np.array[N]): bounds to stay in/out or reach/leave.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the joint positions
                leave the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the joint positions leave the bounds, it results in a success.
            all (bool): this is only used if they are multiple joints. if True, all the joints are checked such that
                they are inside or outside the bounds depending on the other parameters. if False, any joints will be
                checked.
        """
        super(JointPositionCondition, self).__init__(robot, joint_ids=joint_ids, bounds=bounds, out=out, stay=stay,
                                                     all=all)

    def _get_states(self):
        """Get the joint position states."""
        return self.robot.get_joint_positions(self.joints)


class JointVelocityCondition(JointCondition):
    r"""Joint velocity terminal condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the joint velocities are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the joint velocities is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)
    """

    def __init__(self, robot, joint_ids=None, bounds=(None, None), out=False, stay=False, all=False):
        """
        Initialize the joint velocity terminal condition.

        Args:
            robot (Robot): robot instance
            joint_ids (int, int[N], None): joint id or list of joint ids
            bounds (tuple of float / np.array[N]): bounds to stay in/out or reach/leave.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the joint velocities
                leave the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the joint velocities leave the bounds, it results in a success.
            all (bool): this is only used if they are multiple joints. if True, all the joints are checked such that
                they are inside or outside the bounds depending on the other parameters. if False, any joints will be
                checked.
        """
        super(JointVelocityCondition, self).__init__(robot, joint_ids=joint_ids, bounds=bounds, out=out, stay=stay,
                                                     all=all)

    def _get_states(self):
        """Get the joint position states."""
        return self.robot.get_joint_velocities(self.joints)


class JointAccelerationCondition(JointCondition):
    r"""Joint acceleration terminal condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the joint accelerations are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the joint accelerations is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)
    """

    def __init__(self, robot, joint_ids=None, bounds=(None, None), out=False, stay=False, all=False):
        """
        Initialize the joint acceleration terminal condition.

        Args:
            robot (Robot): robot instance
            joint_ids (int, int[N], None): joint id or list of joint ids
            bounds (tuple of float / np.array[N]): bounds to stay in/out or reach/leave.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the joint
                accelerations leave the bounds it results in a failure. if :attr:`stay` is False, it must get outside
                these bounds; if the joint accelerations leave the bounds, it results in a success.
            all (bool): this is only used if they are multiple joints. if True, all the joints are checked such that
                they are inside or outside the bounds depending on the other parameters. if False, any joints will be
                checked.
        """
        super(JointAccelerationCondition, self).__init__(robot, joint_ids=joint_ids, bounds=bounds, out=out, stay=stay,
                                                         all=all)

    def _get_states(self):
        """Get the joint position states."""
        return self.robot.get_joint_accelerations(self.joints)


class JointTorqueCondition(JointCondition):
    r"""Joint torque terminal condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the joint torques are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the joint torques is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)
    """

    def __init__(self, robot, joint_ids=None, bounds=(None, None), out=False, stay=False, all=False):
        """
        Initialize the joint torque terminal condition.

        Args:
            robot (Robot): robot instance
            joint_ids (int, int[N], None): joint id or list of joint ids
            bounds (tuple of float / np.array[N]): bounds to stay in/out or reach/leave.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the joint torques
                leave the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the joint positions leave the bounds, it results in a success.
            all (bool): this is only used if they are multiple joints. if True, all the joints are checked such that
                they are inside or outside the bounds depending on the other parameters. if False, any joints will be
                checked.
        """
        super(JointTorqueCondition, self).__init__(robot, joint_ids=joint_ids, bounds=bounds, out=out, stay=stay,
                                                   all=all)

    def _get_states(self):
        """Get the joint position states."""
        return self.robot.get_joint_torques(self.joints)
