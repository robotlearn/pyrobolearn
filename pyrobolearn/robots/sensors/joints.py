#!/usr/bin/env python
"""Define the joint sensors used in robotics.

This mainly include encoders.
"""

import copy
import time
from abc import ABCMeta
import numpy as np

from pyrobolearn.robots.sensors.sensor import Sensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointSensor(Sensor):
    r"""Joint Sensor (abstract)

    Sensor attached to a joint.
    """
    __metaclass__ = ABCMeta

    def __init__(self, simulator, body_id, joint_ids=None, noise=None, ticks=1, latency=None):
        """Initialize the sensor.

        Args:
            simulator (Simulator): simulator
            body_id (int): unique body id.
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will get all the actuated joints.
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
        """
        super(JointSensor, self).__init__(simulator, body_id=body_id, noise=noise, ticks=ticks, latency=latency)

        # set the joint ids
        if joint_ids is None:
            # get actuated joints
            joint_ids = []
            for joint_id in range(self.sim.num_joints(self.body_id)):
                joint_info = self.sim.get_joint_info(self.body_id, joint_id)
                if joint_info[2] != self.sim.JOINT_FIXED:  # if not a fixed joint
                    joint_ids.append(joint_info[0])
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        elif isinstance(joint_ids, (tuple, list)):
            for i, joint in enumerate(joint_ids):
                if not isinstance(joint, int):
                    raise TypeError("Expecting the given {}th joint id to be an int, instead got: {}".format(i, joint))
        else:
            raise TypeError("Expecting the given 'joint_ids' to be an int or list of int, instead got: "
                            "{}".format(joint_ids))
        self.joint_ids = joint_ids
        self.q_indices = self.sim.get_q_indices(self.body_id, self.joint_ids)

        # joint state
        self._state = {}
        self._prev_state = {}

    #############
    # Operators #
    #############

    def clean(self):
        """Clean sensor values."""
        # update previous and current states
        self._prev_state = self._state
        self._state = {}

    def get_joint_positions(self):
        r"""
        Get the position of the given joint(s).

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[N]: joint positions [rad]
        """
        # check if cached
        if 'q' in self._state:
            # get cached joint positions
            q = self._state['q'][0]
        else:
            # get joint positions and cache it
            q = self.sim.get_joint_positions(self.body_id, self.joint_ids)
            self._state['q'] = [q, time.time()]

        # return joint positions
        return q[self.q_indices]

    def get_joint_velocities(self):
        r"""
        Get the velocity of the given joint(s).

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.array[N]: joint velocities [rad/s]
        """
        # check if cached
        if 'dq' in self._state:
            # get cached joint velocities
            dq = self._state['dq'][0]
        else:
            # get joint velocities and cache it
            dq = self.sim.get_joint_velocities(self.body_id, self.joint_ids)
            self._state['dq'] = [dq, time.time()]

        # return joint velocities
        return dq[self.q_indices]

    def get_joint_accelerations(self):
        r"""
        Get the acceleration of the specified joint(s). If the simulator doesn't provide the joint accelerations, this
        is computed using finite difference :math:`\ddot{q}(t) = \frac{\dot{q}(t) - \dot{q}(t-dt)}{dt}`.

        Warnings: if we use finite difference, note that the first time this method is called, it will return a zero
            vector because we do not have previous joint velocities (i.e. :math:`\dot{q}(t-dt)`) yet.

        Returns:
            if 1 joint:
                float: joint acceleration [rad/s^2]
            if multiple joints:
                np.array[N]: joint accelerations [rad/s^2]
        """
        # check if cached
        if 'ddq' in self._state:
            ddq = self._state[0]
            return ddq[self.q_indices]

        # if simulator supports accelerations
        if self.sim.supports_acceleration():
            return self.sim.get_joint_accelerations(self.body_id, self.joint_ids)

        # else, use finite difference

        # get current joint velocities and time
        if 'dq' in self._state:
            dq, t = self._state['dq']
        else:
            dq, t = self.get_joint_velocities(), time.time()
            self._state['dq'] = [dq, t]

        # if we did not cache the previous joint velocities, return zero vector for accelerations
        if 'dq' not in self._prev_state:
            ddq = np.zeros(len(self.joint_ids))
            self._state['ddq'] = [ddq, t]
            return ddq[self.q_indices]

        # retrieve previous joint velocities and time
        dq_prev, t_prev = self._prev_state['dq']

        # compute time difference
        if self.sim.use_real_time():  # if the simulator is in real-time mode
            dt = (t - t_prev)
        else:  # if we are stepping in the simulator
            dt = self.sim.timestep

        # compute joint accelerations using finite difference, and cache it
        ddq = (dq - dq_prev) / dt
        self._state['ddq'] = [ddq, t]

        # return joint accelerations
        return ddq[self.q_indices]

    def get_joint_torques(self):
        r"""
        Get the applied torque on the given joint(s).

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.array[N]: torques associated to the given joints [Nm]
        """
        return self.sim.get_joint_torques(self.body_id, self.joint_ids)

    def __copy__(self):
        """Return a shallow copy of the sensor. This can be overridden in the child class."""
        return self.__class__(simulator=self.simulator, body_id=self.body_id, joint_id=self.joint_ids,
                              position=self.local_position, orientation=self.local_orientation, rate=self._ticks)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the sensor. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        simulator = memo.get(self.simulator, self.simulator)  # copy.deepcopy(self.simulator, memo)
        body_id = copy.deepcopy(self.body_id)
        joint_id = copy.deepcopy(self.joint_ids)
        position = copy.deepcopy(self.local_position)
        orientation = copy.deepcopy(self.local_orientation)
        sensor = self.__class__(simulator=simulator, body_id=body_id, joint_id=joint_id, position=position,
                                orientation=orientation, rate=self._ticks)
        memo[self] = sensor
        return sensor


class JointEncoderSensor(JointSensor):
    r"""Joint encoder sensor

    The encoder is a sensor that measures rotation allowing to determine the angle, displacement, velocity, or
    acceleration.

    References:
        - [1] Robot encoder: https://www.societyofrobots.com/sensors_encoder.shtml
    """

    def __init__(self, simulator, body_id, joint_ids=None, noise=None, ticks=1, latency=None):
        """
        Initialize the joint encoder sensor.

        Args:
            simulator (Simulator): simulator
            body_id (int): unique body id.
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will get all the actuated joints.
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
        """
        super(JointEncoderSensor, self).__init__(simulator, body_id=body_id, joint_ids=joint_ids, noise=noise,
                                                 ticks=ticks, latency=latency)

        # sense once
        self._sense()

    ##############
    # Properties #
    ##############

    @property
    def q(self):
        """Return the sensed joint position values."""
        return self._data[:len(self.joint_ids)]

    @property
    def dq(self):
        """Return the sensed joint velocity values."""
        return self._data[len(self.joint_ids):2*len(self.joint_ids)]

    @property
    def ddq(self):
        """Return the sensed joint acceleration values."""
        return self._data[2*len(self.joint_ids):]

    ###########
    # Methods #
    ###########

    def get_sensed_joint_positions(self):
        """Return the sensed joint positions."""
        return self.q

    def get_sensed_joint_velocities(self):
        """Return the sensed joint velocities."""
        return self.dq

    def get_sensed_joint_accelerations(self):
        """Return the sensed joint accelerations."""
        return self.ddq

    def _sense(self, apply_noise=True):
        """
        Sense the joint (position, velocities, accelerations) values.

        Args:
            apply_noise (bool): if we should apply the noise or not. Note that the sensor might already have some noise.

        Returns:
            np.array[3*N]: concatenation of joint positions, velocities and accelerations
        """
        # if the simulator supports encoder sensors, return the sensed data
        # if self.simulator.supports_sensors("encoder"):
        #     return self.simulator.get_sensor("encoder", self.body_id, self.joint_ids).sense()

        # joint state #
        positions = self.get_joint_positions()
        velocities = self.get_joint_velocities()
        accelerations = self.get_joint_accelerations()

        # concatenate the data and apply the noise
        data = np.concatenate((positions, velocities, accelerations))
        if apply_noise:
            data = self._noise(data)

        # return the noisy data
        return data
