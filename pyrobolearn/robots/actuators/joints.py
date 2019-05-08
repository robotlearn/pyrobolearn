#!/usr/bin/env python
"""Define the various joint actuators used in robotics.
"""

import copy
import numpy as np

from pyrobolearn.robots.actuators.actuator import Actuator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointActuator(Actuator):
    r"""Joint Actuators

    This defined the joint actuator class; this is an actuator which is attached to a joint and outputs the torque
    to be applied on it using a specific control scheme (e.g. PD control).
    For instance, given a target joint position value, the actuator computes the necessary torque to be applied on
    the joint using a simple PD control (with certain gains).
    """

    def __init__(self, joint_id):
        super(JointActuator, self).__init__()
        self.joint_id = joint_id

    def __copy__(self):
        """Return a shallow copy of the actuator. This can be overridden in the child class."""
        return self.__class__(joint_id=self.joint_id)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the actuator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        joint_id = copy.deepcopy(self.joint_id)
        actuator = self.__class__(joint_id=joint_id)
        memo[self] = actuator
        return actuator


class PDJointActuator(JointActuator):
    r"""PD Joint Actuator

    Compute the torque to be applied on the joint using a PD controller: :math:`\tau = k_p (q_d - q) - k_d \dot{q}`,
    where :math:`q` and :math:`\dot{q}` are the current joint position and velocity respectively, :math:`q_d` is
    the desired joint position, and :math:`k_p` and :math:`k_d` are the PD gains.
    """

    def __init__(self, joint_id, kp=0, kd=0, min_torque=-np.infty, max_torque=np.infty, latency=0):
        super(PDJointActuator, self).__init__(joint_id)
        self.kp = kp
        self.kd = kd
        self.min_torque = min_torque
        self.max_torque = max_torque
        self.latency = latency

    def compute(self, qd, q, dq):
        """
        Compute and return the torque using the PD control scheme.

        Args:
            qd (float): desired joint position
            q (float): current joint position
            dq (float): current joint velocity

        Returns:
            float: computed torque using PD control
        """
        torque = self.kp * (qd - q) - self.kd * dq
        torque = np.clip(torque, self.min_torque, self.max_torque)
        return torque

    def __copy__(self):
        """Return a shallow copy of the actuator. This can be overridden in the child class."""
        return self.__class__(joint_id=self.joint_id, kp=self.kp, kd=self.kd, min_torque=self.min_torque,
                              max_torque=self.max_torque, latency=self.latency)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the actuator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        joint_id = copy.deepcopy(self.joint_id)
        kp = copy.deepcopy(self.kp)
        kd = copy.deepcopy(self.kd)
        min_torque = copy.deepcopy(self.min_torque)
        max_torque = copy.deepcopy(self.max_torque)
        latency = copy.deepcopy(self.latency)
        actuator = self.__class__(joint_id=joint_id, kp=kp, kd=kd, min_torque=min_torque, max_torque=max_torque,
                                  latency=latency)
        memo[self] = actuator
        return actuator


class GearedActuator(JointActuator):
    r"""Geared Actuator
    """

    def __init__(self, joint_id):
        super(GearedActuator, self).__init__(joint_id)


class DirectDriveActuator(JointActuator):
    r"""Direct Drive Actuator
    """

    def __init__(self, joint_id):
        super(DirectDriveActuator, self).__init__(joint_id)


class SEA(JointActuator):
    r"""Series Elastic Actuators

    This actuator has multiple components including springs, gears, encoders, and an electric motors, resulting in
    complex dynamics. Specifically, it is composed of an electric motor, a high gear ratio transmission, an elastic
    element, and two rotary encoders to measure spring deflection and output position. [2]

    References:
        [1] "Series elastic actuators", Pratt et al., 1995
        [2] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
    """

    def __init__(self, joint_id):
        super(SEA, self).__init__(joint_id)


class HydraulicActuator(JointActuator):
    r"""Hydraulic Actuator
    """

    def __init__(self, joint_id):
        super(HydraulicActuator, self).__init__(joint_id)


class JointActuatorApproximator(JointActuator):
    r"""Joint Actuator Approximator.

    This is a joint actuator that uses a function approximator to output the torque values to be applied on the
    actuator given for instance the joint positions. This function approximator has been trained on real data obtained
    from the real actuator and can thus be a better approximation of the way the actual actuator works.
    """

    def __init__(self, joint_id, approximator=None):
        super(JointActuatorApproximator, self).__init__(joint_id)
        self.approximator = approximator

    def __copy__(self):
        """Return a shallow copy of the actuator. This can be overridden in the child class."""
        return self.__class__(joint_id=self.joint_id, approximator=self.approximator)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the actuator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        joint_id = copy.deepcopy(self.joint_id)
        approximator = copy.deepcopy(self.approximator, memo)
        actuator = self.__class__(joint_id=joint_id, approximator=approximator)
        memo[self] = actuator
        return actuator


class ActuatorNet(JointActuatorApproximator):
    r"""Actuator Neural Network

    References:
        [1] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
    """

    def __init__(self, joint_id, nn_model=None):
        super(ActuatorNet, self).__init__(joint_id, approximator=nn_model)


class CoupledJointActuatorApproximator(JointActuator):
    r"""Coupled Joint Actuator Approximator

    Multiple joint ids.
    """
    pass


class CoupledActuatorNet(CoupledJointActuatorApproximator):
    r"""Coupled Actuator Neural Network

    """
    pass
