#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the costs used on link states / actions.
"""

from abc import ABCMeta
import numpy as np

import pyrobolearn as prl
from pyrobolearn.rewards.cost import Cost
from pyrobolearn.utils.transformation import quaternion_distance

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinkCost(Cost):
    r"""(Abstract) Link Cost."""
    __metaclass__ = ABCMeta

    def __init__(self, update_state):
        """
        Initialize the abstract link cost.

        Args:
            update_state (bool): if True it will update the given states before computing the cost.
        """
        super(LinkCost, self).__init__()
        self.update_state = update_state


class DistanceCost(LinkCost):
    """Distance Cost.

    It penalizes the distance between 2 objects. One of the 2 objects must be movable in order for this
    cost to change.

    Mathematically, the cost is given by:

    .. math:: c(p_1, p_2) = - d(p_1, p_2) = - || p_2 - p_1 ||^2

    where :math:`p_1` represents the position of the specified link attached to the first body with respect to a
    frame, and :math:`p_2` represents the position of the specified link attached on the second body with respect to
    that same frame. The distance function used is the Euclidean distance (=L2 norm).
    """

    def __init__(self, body1, body2, link_id1=-1, link_id2=-1, offset=None, update_state=False):
        r"""
        Initialize the distance cost.

        Args:
            body1 (BasePositionState, PositionState, LinkWorldPositionState, Body, Robot): first position state. If
                Body, it will wrap it with a `PositionState`. If Robot, it will wrap it with a `PositionState` or
                `LinkPositionState` depending on the value of :attr:`link_id1`.
            body2 (BasePositionState, PositionState, LinkWorldPositionState, Body, Robot): second position state. If
                Body, it will wrap it with a `PositionState`. If Robot, it will wrap it with a `PositionState` or
                `LinkPositionState` depending on the value of :attr:`link_id1`.
            link_id1 (int): link id associated with the first body that we are interested in. This is only used if
                the given :attr:`body1` is not a state.
            link_id2 (int): link id associated with the second body that we are interested in. This is only used if
                the given :attr:`body2` is not a state.
            offset (None, np.array[3]): 3d offset between body1 and body2.
            update_state (bool): if True it will update the given states before computing the cost.
        """
        super(DistanceCost, self).__init__(update_state=update_state)

        # check body type function
        def check_body_type(body, id_, link_id):
            update_state = False
            if isinstance(body, prl.robots.Body):
                body = prl.states.PositionState(body)
                update_state = True
            elif isinstance(body, prl.robots.Robot):
                if link_id == -1:
                    body = prl.states.PositionState(body)
                else:
                    body = prl.states.LinkWorldPositionState(body, link_ids=link_id)
                update_state = True
            elif not isinstance(body, (prl.states.BasePositionState, prl.states.PositionState,
                                       prl.states.LinkWorldPositionState)):
                raise TypeError("Expecting the given 'body"+str(id_)+"' to be an instance of `Body`, `Robot`, "
                                "`BasePositionState`, `PositionState` or `LinkWorldPositionState`, instead got: "
                                "{}".format(type(body), id_))
            return body, update_state

        self.p1, self.update_state1 = check_body_type(body1, id_=1, link_id=link_id1)
        self.p2, self.update_state2 = check_body_type(body2, id_=2, link_id=link_id2)

    def _compute(self):
        """Compute and return the cost value."""
        if self.update_state1:
            self.p1()
        if self.update_state2:
            self.p2()
        return - np.sum((self.p1.data[0] - self.p2.data[0])**2)


# alias
PositionCost = DistanceCost


class OrientationCost(LinkCost):
    r"""Orientation Cost

    The orientation cost (which uses the distance between two quaternions :math:`q_1` and :math:`q_2`) is given by:

    .. math::

        c(q_1, q_2) = \left\{ \begin{array}{ll}
                2\pi,                           & q1 * \bar{q}_2 = -1 + [0,0,0]^\top \\
                2 || \log(q_1 * \bar{q}_2) ||,  & \text{otherwise}
            \end{array} \right.

    where :math:`*` is the quaternion product, :math:`\bar{q}` is the conjugate of the quaternion,
    :math:`-1 + [0,0,0]^\top` is the only singularity on :math:`\mathbb{S}^3`, and
    :math:`\log: \mathbb{S}^3 \rightarrow \mathbb{R}^3` is the logarithm map.
    """

    def __init__(self, state, target_state, update_state=False):
        """
        Initialize the orientation cost.

        Args:
            state (OrientationState, BaseOrientationState, LinkWorldOrientationState, LinkOrientationState, Body,
                Robot): the orientation state.
            target_state (np.array[4], OrientationState, BaseOrientationState, LinkWorldOrientationState,
                LinkOrientationState, Body, Robot, None): target orientation state. Note that if a np.array is given,
                it will wrap it with the `FixedState`. If None, it will be initialize to the unit quaternion.
            update_state (bool): if True it will update the given states before computing the cost.
        """
        super(OrientationCost, self).__init__(update_state=update_state)
        # TODO
        self.q1 = state
        self.q2 = target_state

    def _compute(self):
        """Compute and return the cost value."""
        return - quaternion_distance(self.q1.data[0], self.q2.data[0])


class LinearVelocityCost(LinkCost):
    r"""Linear velocity cost

    The linear velocity cost is expressed as:

    .. math:: c(v_1, v_2) = || v_2 - v_1 ||^2

    """

    def __init__(self, state, target_state, update_state=False):
        """
        Initialize the linear velocity cost.

        Args:
            state (LinearVelocityState): the linear velocity state.
            target_state (LinearVelocityState, np.array[3], None): target linear velocity state. Note that if a
                np.array is given, it will wrap it with the `FixedState`. If None, it will be initialize to zeros.
            update_state (bool): if True it will update the given states before computing the cost.
        """
        raise NotImplementedError

    def _compute(self):
        """Compute and return the cost value."""
        pass


class AngularVelocityCost(LinkCost):
    r"""Angular velocity cost

    The angular velocity cost is expressed as:

    .. math:: c(\omega_1, \omega_2) = || \omega_2 - \omega_1 ||^2

    """

    def __init__(self, state, target_state, update_state=False):
        """
        Initialize the angular velocity cost.

        Args:
            state (LinearVelocityState): the angular velocity state.
            target_state (LinearVelocityState, np.array[3], None): target angular velocity state. Note that if a
                np.array is given, it will wrap it with the `FixedState`. If None, it will be initialize to zeros.
            update_state (bool): if True it will update the given states before computing the cost.
        """
        raise NotImplementedError

    def _compute(self):
        """Compute and return the cost value."""
        pass
