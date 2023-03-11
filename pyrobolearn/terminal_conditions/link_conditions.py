#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define some link terminal conditions for the environment.
"""

import copy
import numpy as np
from abc import ABCMeta

from pyrobolearn.terminal_conditions.robot_condition import RobotCondition
from pyrobolearn.utils.transformation import get_rpy_from_quaternion


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinkCondition(RobotCondition):
    r"""Link Terminal Condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the dimensions of the link state are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the dimension of the link state is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)

    link states include its position, orientation, or velocity for instance.
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, link_id, wrt_link_id=None, bounds=(None, None), dim=None, out=False, stay=False,
                 all=False):
        """
        Initialize the link terminal condition.

        Args:
            robot (Robot): robot instance.
            link_id (int): link id.
            wrt_link_id (None, int): link id wrt which the link state is based on. if None, the state will be with
                respect to the world frame. If -1, it is wrt the base frame.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the link state
                leaves the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the link state leaves the bounds, it results in a success.
            all (bool): this is only used if they are multiple dimensions. if True, all the dimensions of the state
                are checked if they are inside or outside the bounds depending on the other parameters. if False, any
                dimensions will be checked.
        """
        super(LinkCondition, self).__init__(robot, bounds=bounds, dim=dim, out=out, stay=stay, all=all)

        # set link
        if not isinstance(link_id, int):
            raise TypeError("Expecting the given 'link_id' to be an int, but instead got: {}".format(type(link_id)))
        self.link = link_id

        # set wrt_link_id
        if wrt_link_id is not None and not isinstance(wrt_link_id, int):
            raise TypeError("Expecting the given 'wrt_link_id' to be an int, but instead got: "
                            "{}".format(type(wrt_link_id)))
        self.wrt_link = wrt_link_id

        # check bounds
        self.bounds = self._check_bounds(bounds)

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
        return bounds


class LinkPositionCondition(LinkCondition):
    r"""Link position terminal condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the dimensions of the link position state are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the dimension of the link position state is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)
    """

    def __init__(self, robot, link_id=None, wrt_link_id=None, bounds=(None, None), dim=None, out=False, stay=False,
                 all=False):
        """
        Initialize the link position terminal condition.

        Args:
            robot (Robot): robot instance.
            link_id (int, int[N], None): link id or list of link ids.
            wrt_link_id (None, int): link id wrt which the position is based on. if None, the position will be with
                respect to the world frame. If -1, it is wrt the base frame.
            bounds (tuple of 2 np.array[3] / np.array[N,3], np.array[2,3], np.array[2,N,3]): bounds to stay in/out or
                reach/leave.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the link position
                leaves the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the link position leaves the bounds, it results in a success.
        """
        super(LinkPositionCondition, self).__init__(robot, link_id=link_id, wrt_link_id=wrt_link_id, bounds=bounds,
                                                    dim=dim, out=out, stay=stay, all=all)

    def _get_states(self):
        """Return the link position state."""
        # get the link position
        if self.wrt_link is None:
            position = self.robot.get_link_world_positions(link_ids=self.link, flatten=True)
        else:
            position = self.robot.get_link_positions(link_ids=self.link, wrt_link_id=self.wrt_link, flatten=True)

        if self.dim is None:
            return position
        return position[self.dim]


class LinkOrientationCondition(LinkCondition):
    r"""Link orientation terminal condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the dimensions of the link orientation (expressed as roll-pitch-yaw angles) state are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the dimension of the link orientation (expressed as roll-pitch-yaw angles) state is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)

    Warnings: the orientation is expressed as roll-pitch-yaw angles.
    """

    def __init__(self, robot, link_id, wrt_link_id=None, bounds=(None, None), dim=None, out=False, stay=False,
                 all=False):
        """
        Initialize the link orientation terminal condition.

        Args:
            robot (Robot): robot instance.
            link_id (int, None): link id or list of link ids.
            wrt_link_id (None, int): link id wrt which the orientation is based on. if None, the orientation will be
                with respect to the world frame. If -1, it is wrt the base frame.
            bounds (tuple of 2 float / np.array[3], np.array[2], np.array[2,3]): bounds to stay in/out or reach/leave.
                the orientation is expressed as roll-pitch-yaw angles.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the link
                orientation leaves the bounds it results in a failure. if :attr:`stay` is False, it must get outside
                these bounds; if the link orientation leaves the bounds, it results in a success.
        """
        super(LinkOrientationCondition, self).__init__(robot, link_id=link_id, wrt_link_id=wrt_link_id, bounds=bounds,
                                                       dim=dim, out=out, stay=stay, all=all)

    def _get_states(self):
        """Return the link orientation state."""
        # get the link orientation
        if self.wrt_link is None:
            orientation = self.robot.get_link_world_orientations(link_ids=self.link, flatten=True)
        else:
            orientation = self.robot.get_link_orientations(link_ids=self.link, wrt_link_id=self.wrt_link, flatten=True)

        # convert from quaternion to roll-pitch-yaw angles
        orientation = get_rpy_from_quaternion(orientation)

        # return the proper orientation
        if self.dim is None:
            return orientation
        return orientation[self.dim]


# class LinkVelocityCondition(LinkCondition):
#     r"""Link velocity terminal condition
#     """
#     raise NotImplementedError
