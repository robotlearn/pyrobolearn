#!/usr/bin/env python
"""Define the force torque joint sensors used in robotics.
"""

import numpy as np

from pyrobolearn.robots.sensors.joints import JointSensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointTorqueSensor(JointSensor):
    r"""Joint Torque sensor.

    """

    def __init__(self, simulator, body_id, joint_ids, noise=None, ticks=1, latency=None):
        """
        Initialize the F/T sensor.

        Args:
            simulator (Simulator): simulator
            body_id (int): unique body id.
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will get all the actuated joints.
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
        """
        super(JointTorqueSensor, self).__init__(simulator, body_id=body_id, joint_ids=joint_ids, noise=noise,
                                                ticks=ticks, latency=latency)

        # enable the F/T sensors
        self.sim.enable_joint_force_torque_sensor(body_id, joint_ids=self.joint_ids, enable=True)

    def _sense(self, apply_noise=True):
        """Sense the force/torques values.

        Args:
            apply_noise (bool): if we should apply the noise or not. Note that the sensor might already have some noise.

        Returns:
            np.array[N]: torque values
        """
        # check if the simulator supports that sensor
        if self.sim.supports_sensors("torque"):
            return self.sim.get_sensor("torque", self.body_id, self.joint_ids).sense().reshape(-1)

        torques = self.sim.get_joint_torques(self.body_id, self.joint_ids).reshape(-1)
        if apply_noise:
            torques = self._noise(torques)

        return torques


class JointForceTorqueSensor(JointSensor):
    r"""Joint Force-Torque (FT or F/T) Sensor

    The F/T sensor allows to measure the forces and torques applied to it; "it reports the joint reaction forces in
    the fixed degrees of freedom: a fixed joint will measure all 6DOF joint forces/torques. A revolute joint
    force/torque sensor will measure 5DOF reaction forces along all axis except the revolute axis." [1]

    Note that this work with fixed joints as well.

    References:
        - [1] Pybullet
    """

    def __init__(self, simulator, body_id, joint_ids, noise=None, ticks=1, latency=None):
        """
        Initialize the joint F/T sensor.

        Args:
            simulator (Simulator): simulator
            body_id (int): unique body id.
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will get all the actuated joints.
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
        """
        super(JointForceTorqueSensor, self).__init__(simulator, body_id=body_id, joint_ids=joint_ids, noise=noise,
                                                     ticks=ticks, latency=latency)

        # enable the F/T sensors
        self.sim.enable_joint_force_torque_sensor(body_id, joint_ids=self.joint_ids, enable=True)

    def _sense(self, apply_noise=True):
        """Sense the force/torques values.

        Args:
            apply_noise (bool): if we should apply the noise or not. Note that the sensor might already have some noise.

        Returns:
            np.array[6*N]: F/T values
        """
        # check if the simulator supports that sensor
        if self.sim.supports_sensors("force-torque"):
            return self.sim.get_sensor("force-torque", self.body_id, self.joint_ids).sense().reshape(-1)

        forces = self.sim.get_joint_reaction_forces(self.body_id, self.joint_ids).reshape(-1)
        if apply_noise:
            forces = self._noise(forces)

        return forces
