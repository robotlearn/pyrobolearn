#!/usr/bin/env python
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

# from objective import Objective
from pyrobolearn.rewards.reward import Reward


# class Cost(Objective):
class Cost(Reward):
    r"""Abstract `Cost` class which inherits from the `Objective` class, and is set to be minimized.
    Every classes that defines a cost inherits from this one. A cost is defined as an objective that
    penalizes a certain behavior.
    """
    def __init__(self):
        super(Cost, self).__init__()
        # super(Cost, self).__init__(maximize=False)


def logistic_kernel_function(error, alpha):
    r"""
    The logistic kernel function :math:`K(x|\alpha) = \frac{1}{(e^{\alpha x} + 2 + e^{-\alpha x})} \in [-0.25,0)`.

    Args:
        error (Cost, float): cost (e.g. error term)
        alpha (float): sensitivity

    Return:
        callable, float: logistic kernel function
    """
    if callable(error):
        y = copy.copy(error)  # shallow copy
        y.compute = lambda: 1. / (np.exp(alpha * error() + 2. + np.exp(- alpha * error)))
        return y
    else:
        return 1. / (np.exp(alpha * error) + 2. + np.exp(- alpha * error))


def min_angle_difference(q1, q2):
    r"""
    Return the minimum angle difference between two angles.

    Args:
        q1 (Cost, float): first angle
        q2 (Cost, float): second angle

    Returns:
        callable, float: minimum angle difference
    """
    return


class AngularVelocityErrorCost(Cost):
    r"""Angular Velocity Error Cost

    Return the angular velocity error cost which is defined in [1] as :math:`K(|\hat{\omega} - \omega|, alpha)` where
    :math:`K` is the logistic kernel function given by
    :math:`K(x|\alpha) = \frac{1}{(e^{\alpha x} + 2 + e^{-\alpha x})} \in [-0.25,0)`, with :math:`\alpha > 0.` being
    the sensitivity factor, and :math:`\hat{\omega}` and :math:`\omega` being the target and current angular velocity.

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, angular_velocity_state, target_angular_velocity_state, sensitivity):
        super(AngularVelocityErrorCost, self).__init__()
        self.state = angular_velocity_state
        self.target_state = target_angular_velocity_state
        self.sensitivity = sensitivity

    def compute(self):
        error = np.linalg.norm(self.target_state.data - self.state.data)
        return - logistic_kernel_function(error, self.sensitivity)


class LinearVelocityErrorCost(Cost):
    r"""Linear Velocity Error Cost

    Return the linear velocity error cost which is defined in [1] as :math:`K(|\hat{v} - v|, alpha)` where
    :math:`K` is the logistic kernel function given by
    :math:`K(x|\alpha) = \frac{1}{(e^{\alpha x} + 2 + e^{-\alpha x})} \in [-0.25,0)`, with :math:`\alpha > 0.` being
    the sensitivity factor, and :math:`\hat{v}` and :math:`v` being the target and current linear velocity.

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, velocity_state, velocity_target_state, sensitivity):
        super(LinearVelocityErrorCost, self).__init__()
        self.state = velocity_state
        self.target_state = velocity_target_state
        self.sensitivity = sensitivity

    def compute(self):
        error = np.linalg.norm(self.target_state.data - self.state.data)
        return - logistic_kernel_function(error, self.sensitivity)


class HeightCost(Cost):
    r"""Height Cost

    Height cost defined in [1] as :math:`cost = 1.0` if height < threshold, otherwise 0.

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, height_state, threshold):
        super(HeightCost, self).__init__()
        self.height = height_state
        self.threshold = threshold

    def compute(self):
        if self.height.data < self.threshold:
            return -1.
        return 0


class JointPositionErrorCost(Cost):
    r"""Joint Position Error Cost

    Return the joint position error as defined in [1] as :math:`d(\hat{\phi}, \phi) \in [0, \pi]` where :math:`d(.,.)`
    is the minimum angle difference between two angles, and :math:`\hat{\phi}` and :math:`\phi` are the target and
    current angles.

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, joint_state, target_joint_state):
        super(JointPositionErrorCost, self).__init__()
        self.state = joint_state
        self.target_state = target_joint_state

    def compute(self):
        return - min_angle_difference(self.state.data, self.target_state.data)


class OrientationGravityCost(Cost):
    r"""Orientation Gravity Cost

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, gravity_state, gravity_vector=[0., 0., -1.]):
        super(OrientationGravityCost, self).__init__()
        self.gravity_state = gravity_state
        self.gravity = np.array(gravity_vector)

    def compute(self):
        return np.linalg.norm(self.gravity_state.data - self.gravity)


class TorqueCost(Cost):
    r"""Torque Cost

    Return the cost due to the torques; :math:`cost = ||\tau||^2`.

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self, torque_state):
        super(TorqueCost, self).__init__()
        self.tau = torque_state

    def compute(self):
        return - np.sum(self.tau.data**2)


class PowerCost(Cost):
    r"""Power Consumption Cost

    Return the power consumption cost, where the power is computed as the torque times the velocity.

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self, torque_state, velocity_state):
        super(PowerCost, self).__init__()
        self.tau = torque_state
        self.vel = velocity_state

    def compute(self):
        return - np.sum(np.maximum(self.tau.data * self.vel.data, 0))


class JointAccelerationCost(Cost):
    r"""Joint Acceleration Cost

    Return the joint acceleration cost defined notably in [1] as :math:`cost = ||\ddot{q}||^2`

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self, joint_acceleration_state):
        super(JointAccelerationCost, self).__init__()
        self.ddq = joint_acceleration_state

    def compute(self):
        return - np.sum(self.ddq.data**2)


class JointSpeedCost(Cost):
    r"""Joint Speed Cost

    Return the joint speed cost as computed in [1].

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self, joint_velocity_state, max_joint_speed=None):
        super(JointSpeedCost, self).__init__()
        self.dq = joint_velocity_state
        self.dq_max = max_joint_speed
        if max_joint_speed is None:
            self.dq_max = joint_velocity_state.max

    def compute(self):
        return - np.sum(np.maximum(self.dq_max - np.abs(self.dq.data), 0)**2)


class BodyImpulseCost(Cost):
    r"""Body Impulse Cost

    Return the body impulse cost as computed in [1].

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self, robot):
        super(BodyImpulseCost, self).__init__()
        self.robot = robot

    def compute(self):
        return


class BodySlippageCost(Cost):
    r"""Body Slippage Cost

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
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
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
        [2] Unilateral Contact (Wikipedia): https://en.wikipedia.org/wiki/Unilateral_contact
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
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
        [2] Unilateral Contact (Wikipedia): https://en.wikipedia.org/wiki/Unilateral_contact
    """
    def __init__(self):
        super(FootClearanceCost, self).__init__()

    def compute(self):
        pass


class SelfCollisionCost(Cost):
    r"""Self Collision Cost

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self):
        super(SelfCollisionCost, self).__init__()

    def compute(self):
        pass


class ActionDifferenceCost(Cost):
    r"""Action Difference Cost

    References:
        [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self, action):
        super(ActionDifferenceCost, self).__init__()
        self.action = action

    def compute(self):
        return - (self.action.data - self.action.prev_data)**2


class PhysicsViolationCost(Cost):
    """Physics Violation Cost.

    This cost defines ...
    It was formally defined in [1]. It accepts two arguments.

    [1] 'Automated Discovery and Learning of Complex Movement Behaviors' (PhD thesis), Mordatch, 2015

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


class PowerConsumptionCost(Cost):
    """Power Consumption Cost.

    It penalizes power consumption using u^TWu where W is a weight matrix, and u is the control vector.
    """

    def __init__(self):
        super(PowerConsumptionCost, self).__init__()

    def loss(self, robot):
        return np.dot(robot.getJointTorques(), robot.getJointVelocities())


class DistanceCost(Cost):
    """Distance Cost.

    It penalizes the distance between 2 objects. One of the 2 objects must be movable in order for this
    cost to change.
    """

    def __init__(self):
        super(DistanceCost, self).__init__()

    def loss(self, object1, object2):
        pass


class ImpactCost(Cost):
    """Impact cost.

    Calculates the impact force using the kinetic energy.
    """

    def __init__(self):
        super(ImpactCost, self).__init__()

    def loss(self, object1, object2):
        pass


class DriftCost(Cost):
    """Drift cost.

    Calculates the drift of a moving object wrt a direction.
    """

    def __init__(self):
        super(DriftCost, self).__init__()

    def loss(self, object, direction):
        pass


class ShakeCost(Cost):
    """Shake cost.

    Calculates the
    """
    def __init__(self):
        super(ShakeCost, self).__init__()

    def loss(self, object, direction):
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
