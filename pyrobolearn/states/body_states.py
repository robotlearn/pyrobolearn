#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the various joint states

This includes notably the joint positions, velocities, and force/torque states.
"""

import sys
from abc import ABCMeta, abstractmethod
import numpy as np
import gym

from pyrobolearn.states.state import State
from pyrobolearn.worlds import World
from pyrobolearn.robots import Body
from pyrobolearn.utils.transformation import quaternion_distance

# define long for Python 3.x
if int(sys.version[0]) == 3:
    long = int


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
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
        # check given body
        if not isinstance(body, (Body, int, long)):
            raise TypeError("Expecting an instance of Body, or an identifier from the simulator/world, instead got: "
                            "{}".format(type(body)))
        if isinstance(body, (int, long)):
            if not isinstance(world, World):
                # try to look for the world in global variables
                if 'world' in globals() and isinstance(globals()['world'], World):  # O(1)
                    world = globals()['world']
                else:
                    raise ValueError("When giving the body identifier, the world need to be given as well.")
            body = Body(world.simulator, body)
        self.body = body

        # initialize parent class
        super(BodyState, self).__init__(window_size=window_size, axis=axis, ticks=ticks)

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


class DistanceState(State):
    r"""Distance between two bodies or a link of each body."""

    def __init__(self, body1, body2, link1_id=-1, link2_id=-1, window_size=1, axis=None, ticks=1):
        """
        Initialize the distance state.

        Args:
            body1 (Body): first body instance.
            body2 (Body): second body instance.
            link1_id (int): link id of the first body. By default, it is the base link.
            link2_id (int): link id of the second body. By default, it is the base link.
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
        # check given bodies and links
        if not isinstance(body1, Body) or not isinstance(body2, Body):
            raise TypeError("Expecting instances of Body, instead got: type(body1)={} and "
                            "type(body2)={}".format(type(body1), type(body2)))
        if not isinstance(link1_id, (int, long)) and not isinstance(link2_id, (int, long)):
            raise TypeError("Expecting the link ids to be an int or long, but instead got: "
                            "{}".format(type(link1_id), type(link2_id)))

        # set attributes
        self.body1 = body1
        self.body2 = body2
        self.link1_id = link1_id
        self.link2_id = link2_id
        self.sim = self.body1.simulator

        # initialize parent class
        data = self._read()
        space = gym.spaces.Box(low=-np.infty, high=np.infty, shape=(1,))
        super(DistanceState, self).__init__(data=data, space=space, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next distance state."""
        pos1 = self.sim.get_link_world_positions(body_id=self.body1.id, link_ids=self.link1_id)
        pos2 = self.sim.get_link_world_positions(body_id=self.body2.id, link_ids=self.link2_id)
        return np.linalg.norm(pos1 - pos2)


class OrientationDistanceState(State):
    r"""Distance between the orientation of two bodies, or link of each body.

    This uses the `quaternion_distance` function described in `pyrobolearn.utils.transformation`. For completeness,
    the documentation is reproduced here:

    "Compute the distance metric (on :math:`\mathbb{S}^3`) between two quaternions :math:`q_1` and :math:`q_2`:

    Assuming a quaternion :math:`q` is represented as :math:`s + \pmb{v}` where :math:`s \in \mathbb{R}` is the scalar
    part and :math:`\pmb{v} \in \mathbb{R}^3` is the vector part, the distance is given by:

    .. math::

        d(q_1, q_2) = \left\{ \begin{array}{ll}
                2\pi, & q1 * \bar{q}_2 = -1 + [0,0,0]^\top \\
                2 || \log(q_1 * \bar{q}_2) ||, & \text{otherwise}
            \end{array} \right.

    where :math:`-1 + [0,0,0]^\top` is the only singularity on :math:`\mathbb{S}^3`.

    Note that this distance is not a metric on :math:`SO(3)` (the set of all orientations, which is by the way not a
    vector space but a group and a real 3d manifold)."

    """

    def __init__(self, body1, body2, link1_id=-1, link2_id=-1, window_size=1, axis=None, ticks=1):
        """
        Initialize the orientation distance state.

        Args:
            body1 (Body): first body instance.
            body2 (Body): second body instance.
            link1_id (int): link id of the first body. By default, it is the base link.
            link2_id (int): link id of the second body. By default, it is the base link.
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
        # check given bodies and links
        if not isinstance(body1, Body) or not isinstance(body2, Body):
            raise TypeError("Expecting instances of Body, instead got: type(body1)={} and "
                            "type(body2)={}".format(type(body1), type(body2)))
        if not isinstance(link1_id, (int, long)) and not isinstance(link2_id, (int, long)):
            raise TypeError("Expecting the link ids to be an int or long, but instead got: "
                            "{}".format(type(link1_id), type(link2_id)))

        # set attributes
        self.body1 = body1
        self.body2 = body2
        self.link1_id = link1_id
        self.link2_id = link2_id
        self.sim = self.body1.simulator

        # initialize parent class
        data = self._read()
        space = gym.spaces.Box(low=0, high=2*np.pi, shape=(1,))
        super(OrientationDistanceState, self).__init__(data=data, space=space, window_size=window_size, axis=axis,
                                                       ticks=ticks)

    def _read(self):
        """Read the next distance state."""
        q1 = self.sim.get_link_world_orientations(body_id=self.body1.id, link_ids=self.link1_id)
        q2 = self.sim.get_link_world_orientations(body_id=self.body2.id, link_ids=self.link2_id)
        return quaternion_distance(q1, q2)
