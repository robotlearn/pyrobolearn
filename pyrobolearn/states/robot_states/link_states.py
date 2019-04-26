#!/usr/bin/env python
"""Define the various link states

This includes notably the link positions and velocities.
"""

from abc import ABCMeta

from pyrobolearn.states.robot_states.robot_states import RobotState, Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
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
        self.data = self.robot.get_link_positions(self.links)


class LinkOrientationState(LinkState):
    r"""Link Orientation state
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
        super(LinkOrientationState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):  # TODO: convert
        """Read the next link orientation state."""
        self.data = self.robot.get_link_orientations(self.links)


class LinkVelocityState(LinkState):
    r"""Link velocity state
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
        super(LinkVelocityState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next link velocity state."""
        self.data = self.robot.get_link_velocities(self.links)


class LinkLinearVelocityState(LinkState):
    r"""Link linear velocity state
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
        super(LinkLinearVelocityState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next link linear velocity state."""
        self.data = self.robot.get_link_linear_velocities(self.links)


class LinkAngularVelocityState(LinkState):
    r"""Link angular velocity state
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
        super(LinkAngularVelocityState, self).__init__(robot, link_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next link angular velocity state."""
        self.data = self.robot.get_link_angular_velocities(self.links)
