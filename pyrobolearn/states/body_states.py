#!/usr/bin/env python
"""Define the various joint states

This includes notably the joint positions, velocities, and force/torque states.
"""

from abc import ABCMeta, abstractmethod

from pyrobolearn.states.state import State
from pyrobolearn.worlds import World
from pyrobolearn.robots import Body


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BodyState(State):
    """Body state (abstract)
    """
    __metaclass__ = ABCMeta

    def __init__(self, body, world=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the body state.

        Args:
            body (Body, int): body or unique body id.
            world (None, World): world instance if the body id was given.
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

        super(BodyState, self).__init__(window_size=window_size, axis=axis, ticks=ticks)
        if not isinstance(body, (Body, int)):
            raise TypeError("Expecting an instance of Body, or an identifier from the simulator/world.")
        if isinstance(body, int):
            if not isinstance(world, World):
                # try to look for the world in global variables
                if 'world' in globals() and isinstance(globals()['world'], World):  # O(1)
                    world = globals()['world']
                else:
                    raise ValueError("When giving the body identifier, the world need to be given as well.")
            body = Body(world.simulator, body)
        self.body = body

    @abstractmethod
    def _read(self):
        pass


class PositionState(BodyState):
    """Position of a body in the world.
    """

    def __init__(self, body, world=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the position state.

        Args:
            body (Body, int): body or unique body id.
            world (None, World): world instance if the body id was given.
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
        super(PositionState, self).__init__(body, world, window_size=window_size, axis=axis, ticks=ticks)
        self.data = self.body.position

    def _read(self):
        """Read the next body position state."""
        self.data = self.body.position


class OrientationState(BodyState):
    """Orientation of a body in the world.
    """

    def __init__(self, body, world=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the orientation state.

        Args:
            body (Body, int): body or unique body id.
            world (None, World): world instance if the body id was given.
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
        super(OrientationState, self).__init__(body, world, window_size=window_size, axis=axis, ticks=ticks)
        self.data = self.body.orientation

    def _read(self):
        """Read the next body orientation."""
        self.data = self.body.orientation


class VelocityState(BodyState):
    """Velocity of a body in the world
    """

    def __init__(self, body, world=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the velocity state.

        Args:
            body (Body, int): body or unique body id.
            world (None, World): world instance if the body id was given.
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
        super(VelocityState, self).__init__(body, world, window_size=window_size, axis=axis, ticks=ticks)
        self.data = self.body.velocity

    def _read(self):
        """Read the next body velocity state."""
        self.data = self.body.velocity
