#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the various link states

This includes notably the link positions and velocities.
"""

import copy
from abc import ABCMeta

from pyrobolearn.states.robot_states.robot_states import RobotState, Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinkState(RobotState):
    r"""Link state of a robot
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, link_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        # check if robot instance
        if not isinstance(robot, Robot):
            raise TypeError("The 'robot' parameter has to be an instance of Robot")

        # get links from robot
        if link_ids is None:
            link_ids = range(robot.num_links)
        self.links = link_ids

        # call parent constructor
        super(LinkState, self).__init__(robot, window_size=window_size, axis=axis, ticks=ticks)

    def __copy__(self):
        """Return a shallow copy of the state. This can be overridden in the child class."""
        return self.__class__(robot=self.robot, link_ids=self.links, window_size=self.window_size, axis=self.axis,
                              ticks=self.ticks)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the state. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        link_ids = copy.deepcopy(self.links)
        state = self.__class__(robot=robot, link_ids=link_ids, window_size=self.window_size, axis=self.axis,
                               ticks=self.ticks)

        memo[self] = state
        return state


class LinkPositionState(LinkState):
    r"""Link Position state
    """

    def __init__(self, robot, link_ids=None, wrt_link_id=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link position state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            wrt_link_id (int, None): link with respect to which we compute the position of the other links. If None,
                it will be the base.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        self.wrt_link_id = wrt_link_id
        super(LinkPositionState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next link position state."""
        return self.robot.get_link_positions(self.links, wrt_link_id=self.wrt_link_id)


class LinkWorldPositionState(LinkState):
    r"""Link World Position state
    """

    def __init__(self, robot, link_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link world position state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(LinkWorldPositionState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next link world position state."""
        self.data = self.robot.get_link_world_positions(self.links, flatten=True)


class LinkOrientationState(LinkState):
    r"""Link Orientation state
    """

    def __init__(self, robot, link_ids=None, wrt_link_id=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link world orientation state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            wrt_link_id (int, None): link with respect to which we compute the position of the other links. If None,
                it will be the base.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        self.wrt_link_id = wrt_link_id
        super(LinkOrientationState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):  # TODO: convert
        """Read the next link orientation state."""
        self.data = self.robot.get_link_orientations(self.links, wrt_link_id=self.wrt_link_id, flatten=True)


class LinkWorldOrientationState(LinkState):
    r"""Link World Orientation state
    """

    def __init__(self, robot, link_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link world orientation state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(LinkWorldOrientationState, self).__init__(robot, link_ids, window_size=window_size, axis=axis,
                                                        ticks=ticks)

    def _read(self):  # TODO: convert
        """Read the next link world orientation state."""
        self.data = self.robot.get_link_world_orientations(self.links, flatten=True)


class LinkVelocityState(LinkState):
    r"""Link velocity state
    """

    def __init__(self, robot, link_ids=None, wrt_link_id=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link velocity state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            wrt_link_id (int, None): link with respect to which we compute the position of the other links. If None,
                it will be the base.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        self.wrt_link_id = wrt_link_id
        super(LinkVelocityState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next link velocity state."""
        self.data = self.robot.get_link_velocities(self.links, wrt_link_id=self.wrt_link_id, flatten=True)


class LinkWorldVelocityState(LinkState):
    r"""Link world velocity state
    """

    def __init__(self, robot, link_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link velocity state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(LinkWorldVelocityState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next link world velocity state."""
        self.data = self.robot.get_link_world_velocities(self.links, flatten=True)


class LinkLinearVelocityState(LinkState):
    r"""Link linear velocity state
    """

    def __init__(self, robot, link_ids=None, wrt_link_id=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link linear velocity state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            wrt_link_id (int, None): link with respect to which we compute the position of the other links. If None,
                it will be the base.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        self.wrt_link_id = wrt_link_id
        super(LinkLinearVelocityState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next link linear velocity state."""
        self.data = self.robot.get_link_linear_velocities(self.links, wrt_link_id=self.wrt_link_id, flatten=True)


class LinkWorldLinearVelocityState(LinkState):
    r"""Link world linear velocity state
    """

    def __init__(self, robot, link_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link linear velocity state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(LinkWorldLinearVelocityState, self).__init__(robot, link_ids, window_size=window_size, axis=axis,
                                                           ticks=ticks)

    def _read(self):
        """Read the next link world linear velocity state."""
        self.data = self.robot.get_link_world_linear_velocities(self.links, flatten=True)


class LinkAngularVelocityState(LinkState):
    r"""Link angular velocity state
    """

    def __init__(self, robot, link_ids=None, wrt_link_id=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link angular velocity state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            wrt_link_id (int, None): link with respect to which we compute the position of the other links. If None,
                it will be the base.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        self.wrt_link_id = wrt_link_id
        super(LinkAngularVelocityState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next link angular velocity state."""
        self.data = self.robot.get_link_angular_velocities(self.links, wrt_link_id=self.wrt_link_id, flatten=True)


class LinkWorldAngularVelocityState(LinkState):
    r"""Link world angular velocity state
    """

    def __init__(self, robot, link_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the link angular velocity state.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(LinkWorldAngularVelocityState, self).__init__(robot, link_ids, window_size=window_size, axis=axis,
                                                            ticks=ticks)

    def _read(self):
        """Read the next link world angular velocity state."""
        self.data = self.robot.get_link_world_angular_velocities(self.links, flatten=True)
