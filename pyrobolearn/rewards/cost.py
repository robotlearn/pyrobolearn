#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define most common costs used in reinforcement learning, control, and optimization.

A cost is defined as an objective that penalizes a certain behavior.
The `Cost` class inherits from the `Objective` class.

To see the documentation of a certain cost in the python interpreter, just type:
```python
from reward import <Cost>
print(<Cost>.__doc__)
```

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

import copy
import numpy as np
import torch

import pyrobolearn as prl
from pyrobolearn.robots.robot import Robot
# from objective import Objective
import pyrobolearn.states as states
import pyrobolearn.actions as actions
from pyrobolearn.rewards.reward import Reward


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# class Cost(Objective):
class Cost(Reward):
    r"""Abstract `Cost` class which inherits from the `Objective` class, and is set to be minimized.
    Every classes that defines a cost inherits from this one. A cost is defined as an objective that
    penalizes a certain behavior.
    """
    def __init__(self, state=None, action=None, costs=None, range=(-np.infty, np.infty)):
        super(Cost, self).__init__(state=state, action=action, rewards=costs, range=range)
        # super(Cost, self).__init__(maximize=False)


def logistic_kernel_function(error, alpha):
    r"""
    The logistic kernel function :math:`K(x|\alpha) = \frac{1}{(e^{\alpha x} + 2 + e^{-\alpha x})} \in [-0.25,0)`,
    where :math:`x` is an error term, and :math:`\alpha` is a sensitivity factor.

    According to the authors of [1,2]: "We found the logistic kernel function to be more useful than Euclidean norm,
    which is a more common choice. An Euclidean norm generates a high cost in the beginning of training where the
    tracking error is high such that termination (i.e. falling) becomes more rewarding strategy. On the other hand,
    the logistic kernel ensures that the cost is lower-bounded by zero and termination becomes less favorable. Many
    other bell-shaped kernels (Gaussian, triweight, biweight, etc) have the same functionality and can be used instead
    of a logistic kernel."

    Args:
        error (Cost, float): cost (e.g. error term)
        alpha (float): sensitivity

    Return:
        callable, float: logistic kernel function

    References:
        - [1] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
        - [2] Supp: https://robotics.sciencemag.org/content/robotics/suppl/2019/01/14/4.26.eaau5872.DC1/aau5872_SM.pdf
    """
    if callable(error):
        y = copy.copy(error)  # shallow copy
        y.compute = lambda: 1. / (np.exp(alpha * error() + 2. + np.exp(- alpha * error)))
        return y
    else:
        return 1. / (np.exp(alpha * error) + 2. + np.exp(- alpha * error))


class AngularVelocityErrorCost(Cost):
    r"""Angular Velocity Error Cost

    Return the angular velocity error cost which is defined in [1] as :math:`K(|\hat{\omega} - \omega|, alpha)` where
    :math:`K` is the logistic kernel function given by
    :math:`K(x|\alpha) = \frac{1}{(e^{\alpha x} + 2 + e^{-\alpha x})} \in [-0.25,0)`, with :math:`\alpha > 0.` being
    the sensitivity factor, and :math:`\hat{\omega}` and :math:`\omega` being the target and current angular velocity.

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, angular_velocity_state, target_angular_velocity_state, sensitivity):
        super(AngularVelocityErrorCost, self).__init__()
        self.state = angular_velocity_state
        self.target_state = target_angular_velocity_state
        self.sensitivity = sensitivity

    def _compute(self):
        error = np.linalg.norm(self.target_state.data - self.state.data)
        return - logistic_kernel_function(error, self.sensitivity)


class LinearVelocityErrorCost(Cost):
    r"""Linear Velocity Error Cost

    Return the linear velocity error cost which is defined in [1] as :math:`K(|\hat{v} - v|, alpha)` where
    :math:`K` is the logistic kernel function given by
    :math:`K(x|\alpha) = \frac{1}{(e^{\alpha x} + 2 + e^{-\alpha x})} \in [-0.25,0)`, with :math:`\alpha > 0.` being
    the sensitivity factor, and :math:`\hat{v}` and :math:`v` being the target and current linear velocity.

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, velocity_state, velocity_target_state, sensitivity):
        super(LinearVelocityErrorCost, self).__init__()
        self.state = velocity_state
        self.target_state = velocity_target_state
        self.sensitivity = sensitivity

    def _compute(self):
        error = np.linalg.norm(self.target_state.data - self.state.data)
        return - logistic_kernel_function(error, self.sensitivity)


class HeightCost(Cost):
    r"""Height Cost

    Height cost defined in [1] as :math:`cost = 1.0` if height < threshold, otherwise 0.

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, height_state, threshold):
        super(HeightCost, self).__init__()
        self.height = height_state
        self.threshold = threshold

    def _compute(self):
        if self.height.data < self.threshold:
            return -1.
        return 0


class OrientationGravityCost(Cost):
    r"""Orientation Gravity Cost

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, state, gravity_vector=[0., 0., -1.]):
        super(OrientationGravityCost, self).__init__()
        self.gravity_state = state
        self.gravity = np.asarray(gravity_vector)

    def _compute(self):
        return np.linalg.norm(self.gravity_state.data[0] - self.gravity)


class PowerCost(Cost):
    r"""Power Consumption Cost

    Return the power consumption cost, where the power is computed as the torque times the velocity. This is given by
    [1] as:

    .. math:: \sum

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, torque_state, velocity_state):
        super(PowerCost, self).__init__()
        self.tau = torque_state
        self.vel = velocity_state

    def _compute(self):
        return - np.sum(np.maximum(self.tau.data[0] * self.vel.data[0], 0))


class PowerAbsoluteCost(Cost):
    r"""Power Consumption Cost

    """

    def __init__(self, torque_state, velocity_state):
        super(PowerCost, self).__init__()
        self.tau = torque_state
        self.vel = velocity_state

    def _compute(self):
        return - np.sum(np.maximum(self.tau.data[0] * self.vel.data[0], 0))


class BodyImpulseCost(Cost):
    r"""Body Impulse Cost

    Return the body impulse cost as computed in [1].

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self, robot):
        super(BodyImpulseCost, self).__init__()
        self.robot = robot

    def compute(self):
        pass


class BodySlippageCost(Cost):
    r"""Body Slippage Cost

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self):
        super(BodySlippageCost, self).__init__()

    def compute(self):
        pass


class FootSlippageCost(Cost):
    r"""Foot Slippage Cost

    'In mechanics, a unilateral contact denotes a mechanical constraint which prevents penetration between two bodies.
    These bodies may be rigid or flexible. A unilateral contact is usually associated with a gap function g which
    measures the distance between the two bodies and a contact force' [2]

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
        - [2] Unilateral Contact (Wikipedia): https://en.wikipedia.org/wiki/Unilateral_contact
    """
    def __init__(self):
        super(FootSlippageCost, self).__init__()

    def compute(self):
        pass


class FootClearanceCost(Cost):
    r"""Foot Clearance Cost

    'In mechanics, a unilateral contact denotes a mechanical constraint which prevents penetration between two bodies.
    These bodies may be rigid or flexible. A unilateral contact is usually associated with a gap function g which
    measures the distance between the two bodies and a contact force' [2]

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
        - [2] Unilateral Contact (Wikipedia): https://en.wikipedia.org/wiki/Unilateral_contact
    """
    def __init__(self):
        super(FootClearanceCost, self).__init__()

    def compute(self):
        pass


class SelfCollisionCost(Cost):
    r"""Self Collision Cost

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self):
        super(SelfCollisionCost, self).__init__()

    def compute(self):
        pass


class PhysicsViolationCost(Cost):
    """Physics Violation Cost.

    This cost defines ...
    It was formally defined in [1]. It accepts two arguments.

    References:
    - [1] 'Automated Discovery and Learning of Complex Movement Behaviors' (PhD thesis), Mordatch, 2015

    .. seealso: `cio.py` in 'pyrobolearn/optim' which uses this cost.
    """

    def __init__(self):
        super(PhysicsViolationCost, self).__init__()


class ContactInvariantCost(Cost):
    """Contact Invariant Cost.

    This cost defines ... and is used in Contact Invariant Optimization (CIO).
    It was formally defined in [1]. It accepts two arguments.

    [1] 'Automated Discovery and Learning of Complex Movement Behaviors' (PhD thesis), Mordatch, 2015

    .. seealso: `cio.py` in 'pyrobolearn/optim' which uses this cost.
    """

    def __init__(self):
        super(ContactInvariantCost, self).__init__()


class ImpactCost(Cost):
    """Impact cost.

    Calculates the impact force using the kinetic energy.
    """

    def __init__(self, body1, body2):
        super(ImpactCost, self).__init__()

    def _compute(self):
        pass


class SpeedCost(Cost):
    """Speed cost.

    """
    def __init__(self):
        pass


class JerkCost(Cost):
    """Jerk cost.

    Calculates the jerk of an object.
    """
    def __init__(self, object, dt):
        self.object = object
        self.dt = dt


class ZMPCost(Cost):
    """ZMP Cost.
    """
    def __init__(self):
        super(ZMPCost, self).__init__()
