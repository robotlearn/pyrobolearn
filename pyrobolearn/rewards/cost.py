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

import pyrobolearn as prl
from pyrobolearn.robots.robot import Robot
# from objective import Objective
import pyrobolearn.states as states
import pyrobolearn.actions as actions
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
        q1 (float, np.array[N]): first angle(s)
        q2 (float, np.array[N]): second angle(s)

    Returns:
        float, np.array[N]: minimum angle difference(s)
    """
    diff = np.maximum(q1, q2) - np.minimum(q1, q2)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff


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


class JointPositionErrorCost(Cost):
    r"""Joint Position Error Cost

    Return the joint position error as defined in [1] as :math:`d(\hat{\phi}, \phi) \in [0, \pi]` where :math:`d(.,.)`
    is the minimum angle difference between two angles, and :math:`\hat{\phi}` and :math:`\phi` are the target and
    current angles.

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, joint_state, target_joint_state):
        super(JointPositionErrorCost, self).__init__()
        self.state = joint_state
        self.target_state = target_joint_state

    def _compute(self):
        return - min_angle_difference(self.state.data[0], self.target_state.data[0])


class OrientationGravityCost(Cost):
    r"""Orientation Gravity Cost

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, gravity_state, gravity_vector=[0., 0., -1.]):
        super(OrientationGravityCost, self).__init__()
        self.gravity_state = gravity_state
        self.gravity = np.array(gravity_vector)

    def _compute(self):
        return np.linalg.norm(self.gravity_state.data - self.gravity)


class JointPositionCost(Cost):
    r"""Joint Position Cost

    Return the cost such that measures the L2 norm between the current joint positions and the target joint positions:
    :math:`||d(q_{target},q)||^2` where :math:`d(\cdot, \cdot) \in [-\pi, \pi]` is the minimum distance between two
    angles.
    """

    def __init__(self, joint_position_state, target_joint_position):
        r"""
        Initialize the joint position cost.

        Args:
            joint_position_state (JointPositionState): joint position state.
            target_joint_position (np.array[N]): target joint positions.
        """
        super(JointPositionCost, self).__init__()
        if not isinstance(joint_position_state, prl.states.JointPositionState):
            raise TypeError("Expecting the given 'joint_position_state' to be an instance of `JointPositionState`, "
                            "but instead got: {}".format(type(joint_position_state)))
        self.q = joint_position_state
        self.target_q = target_joint_position

    def _compute(self):
        """Compute and return the cost value."""
        return - np.sum(min_angle_difference(self.q.data[0], self.target_q)**2)


class JointVelocityCost(Cost):
    r"""Joint Velocity Cost

    Return the cost due to the joint velocities: :math:`|| \dot{q} ||^2`
    """
    def __init__(self, joint_velocity_state):
        """
        Initialize the joint velocity cost.

        Args:
            joint_velocity_state (JointVelocityState): joint velocity state.
        """
        super(JointVelocityCost, self).__init__()
        if not isinstance(joint_velocity_state, prl.states.JointVelocityState):
            raise TypeError("Expecting the given 'joint_velocity_state' to be an instance of `JointVelocityState`, "
                            "but instead got: {}".format(type(joint_velocity_state)))
        self.dq = joint_velocity_state

    def _compute(self):
        """Compute and return the cost value."""
        return - np.sum(self.dq.data[0] ** 2)


class JointTorqueCost(Cost):
    r"""Torque Cost

    Return the cost due to the joint torques; :math:`|| \tau ||^2`.
    """

    def __init__(self, joint_torque_state):
        """
        Initialize the joint torque cost.

        Args:
            joint_torque_state (JointForceTorqueState): joint torque state.
        """
        super(JointTorqueCost, self).__init__()
        if not isinstance(joint_torque_state, prl.states.JointForceTorqueState):
            raise TypeError("Expecting the given 'joint_torque_state' to be an instance of `JointForceTorqueState`, "
                            "but instead got: {}".format(type(joint_torque_state)))
        self.tau = joint_torque_state

    def _compute(self):
        """Compute and return the cost value."""
        return - np.sum(self.tau.data[0]**2)


class PowerCost(Cost):
    r"""Power Consumption Cost

    Return the power consumption cost, where the power is computed as the torque times the velocity.

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """
    def __init__(self, torque_state, velocity_state):
        super(PowerCost, self).__init__()
        self.tau = torque_state
        self.vel = velocity_state

    def compute(self):
        return - np.sum(np.maximum(self.tau.data[0] * self.vel.data[0], 0))


class JointPowerConsumptionCost(Cost):
    r"""Joint Power Consumption Cost

    Return the joint power consumption cost, where the power is computed as the torque times the velocity.
    """

    def __init__(self, state, joint_ids=None, update_state=False):
        """
        Initialize the Joint Power Consumption cost.

        Args:
            torque (Robot, State): robot instance, or the state. The state must contains the `JointForceTorqueState`
                and the `JointVelocityState`.
            joint_ids (None, int, list of int): joint ids. This used if `torque` is a `Robot` instance.
            update_state (bool): If True, it will update the state.
        """
        self.update_state = update_state
        if isinstance(state, Robot):
            torque = states.JointForceTorqueState(state, joint_ids=joint_ids)
            velocity = states.JointVelocityState(state, joint_ids=joint_ids)
            self.update_state = True
            # state = torque + velocity
        elif isinstance(state, states.State):
            # check if they have the correct state
            torque = state.lookfor(states.JointForceTorqueState)
            if torque is None:
                raise ValueError("Didn't find a `JointForceTorqueState` instance in the given states.")
            velocity = state.lookfor(states.JointVelocityState)
            if velocity is None:
                raise ValueError("Didn't find a `JointVelocityState` instance in the given states.")
        else:
            raise TypeError("Expecting the state to be an instance of `State` or `Robot`.")
        super(JointPowerConsumptionCost, self).__init__()  # state=state)
        self.tau = torque
        self.vel = velocity
        self.update_state = update_state

    def compute(self):
        if self.update_state:
            self.tau()
            self.vel()
        return - np.sum((self.tau.data[0] * self.vel.data[0])**2)


class JointAccelerationCost(Cost):
    r"""Joint Acceleration Cost

    Return the joint acceleration cost defined notably in [1] as :math:`cost = ||\ddot{q}||^2`

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
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
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
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


class ActionDifferenceCost(Cost):
    r"""Action Difference Cost

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
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


class DistanceCost(Cost):
    """Distance Cost.

    It penalizes the distance between 2 objects. One of the 2 objects must be movable in order for this
    cost to change.

    Mathematically, the cost is given by:

    .. math:: c(l1, l2) = d(l1, l2) = - || l1 - l2 ||_2

    where :math:`l1` represents a link attached to the first body, and :math:`l2` represents a link attached on the
    second body. The distance function used is the Euclidean distance (=L2 norm).
    """

    def __init__(self, body1, body2, link_id1=-1, link_id2=-1, offset=None):
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
        """
        super(DistanceCost, self).__init__()

        def check_body_type(body, id_, link_id):
            update_state = False
            if isinstance(body, prl.robots.Body):
                body = states.PositionState(body)
                update_state = True
            elif isinstance(body, Robot):
                if link_id == -1:
                    body = states.PositionState(body)
                else:
                    body = states.LinkWorldPositionState(body, link_ids=link_id)
                update_state = True
            elif not isinstance(body, (states.BasePositionState, states.PositionState, states.LinkWorldPositionState)):
                raise TypeError("Expecting the given 'body"+str(id_)+"' to be an instance of `Body`, `Robot`, "
                                "`BasePositionState`, `PositionState` or `LinkWorldPositionState`, instead got: "
                                "{}".format(type(body), id_))
            return body, update_state

        self.body1, self.update_state1 = check_body_type(body1, id_=1, link_id=link_id1)
        self.body2, self.update_state2 = check_body_type(body2, id_=2, link_id=link_id2)

    def compute(self):
        if self.update_state1:
            self.body1()
        if self.update_state2:
            self.body2()
        p1 = self.body1.data[0]
        p2 = self.body2.data[0]
        # print("P1: {}".format(p1))
        # print("P2: {}".format(p2))
        return - np.linalg.norm(p1 - p2)


class ImpactCost(Cost):
    """Impact cost.

    Calculates the impact force using the kinetic energy.
    """

    def __init__(self):
        super(ImpactCost, self).__init__()

    def compute(self, object1, object2):
        pass


class DriftCost(Cost):
    """Drift cost.

    Calculates the drift of a moving object wrt a direction.
    """

    def __init__(self):
        super(DriftCost, self).__init__()

    def compute(self, object, direction):
        pass


class ShakeCost(Cost):
    """Shake cost.

    Calculates the
    """
    def __init__(self):
        super(ShakeCost, self).__init__()

    def compute(self, object, direction):
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
