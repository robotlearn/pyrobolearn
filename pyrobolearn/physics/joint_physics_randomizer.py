#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the `JointPhysicsRandomizer` class which randomizes the physical attributes / properties of a joint or
multiple joints of a specific body.

Dependencies:
- `pyrobolearn.physics`
"""

import collections.abc
import numpy as np

from pyrobolearn.physics.body_physics_randomizer import BodyPhysicsRandomizer
from pyrobolearn.robots.robot import Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointPhysicsRandomizer(BodyPhysicsRandomizer):
    r"""Joint Physics Randomizer

    The joint physics randomizer can randomize the physical attributes of a joint. For instance, this can be the
    joint friction or damping coefficients.
    """

    def __init__(self, robot, joint_ids=None, joint_frictions=None, joint_dampings=None):
        """
        Initialize the joint physics randomizer.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id(s).
            joint_frictions (float, list/tuple of float, np.array[N], None): joint friction bounds. If None, it
                doesn't randomize this parameter.
            joint_dampings (float, list/tuple of float, np.array[N], None): joint damping bounds. If None, it doesn't
                randomize this parameter.
        """
        if not isinstance(robot, Robot):
            raise TypeError("Expecting the given 'robot' to be an instance of `Robot`, instead got: "
                            "{}".format(type(robot)))
        super(JointPhysicsRandomizer, self).__init__(robot)

        # set joint ids
        self.joints = joint_ids

        # set bounds
        self.joint_friction_bounds = joint_frictions
        self.joint_damping_bounds = joint_dampings

    ##############
    # Properties #
    ##############

    @property
    def joints(self):
        """Return the list of joint ids."""
        return self._joints

    @joints.setter
    def joints(self, joints):
        """Set the joint id or the list of joint ids."""
        if joints is None:
            joints = self.body.joints
        elif isinstance(joints, int):
            joints = [joints]
        elif isinstance(joints, collections.abc.Iterable):
            for idx, joint in enumerate(joints):
                if not isinstance(joint, int):
                    raise TypeError("The {} element of the given list of joints is not an integer, instead got: "
                                    "{}".format(idx, type(joint)))
        else:
            raise TypeError("Expecting the given joints to be an integer or a list of integers, instead got: "
                            "{}".format(type(joints)))
        self._joints = joints

    @property
    def joint_frictions(self):
        """Return the joint friction associated with the joints."""
        return self.body.get_joint_frictions(joint_ids=self.joints)

    @joint_frictions.setter
    def joint_frictions(self, values):
        """Set the given joint friction values to each joint."""
        raise NotImplementedError("This feature is not available yet.")

    @property
    def joint_friction_bounds(self):
        """Return the joint friction bounds."""
        return self._joint_friction_bounds

    @joint_friction_bounds.setter
    def joint_friction_bounds(self, values):
        """Set the given joint frictions bounds."""
        self._check_bounds('joint_frictions', values)
        self._joint_friction_bounds = values

    @property
    def joint_dampings(self):
        """Return the joint dampings associated with the joints."""
        return self.body.get_joint_dampings(self.joints)

    @joint_dampings.setter
    def joint_dampings(self, values):
        """Set the given joint damping values to each joint."""
        # check values
        if isinstance(values, (float, int)):
            values = [values] * len(self.joints)
        elif len(values) != len(self.joints):
            raise ValueError("The number of given joint damping values (={}) does not match with the number of "
                             "joints (={})".format(len(values), len(self.joints)))

        # set the joint dampings
        for joint, value in zip(self.joints, values):
            self.simulator.change_dynamics(body_id=self.body.id, link_id=joint, joint_damping=value)

    @property
    def joint_damping_bounds(self):
        """Return the joint damping bounds."""
        return self._joint_damping_bounds

    @joint_damping_bounds.setter
    def joint_damping_bounds(self, values):
        """Set the given joint damping bounds."""
        self._check_bounds('joint_dampings', values)
        self._joint_damping_bounds = values

    ###########
    # Methods #
    ###########

    def names(self):
        """Return an iterator over the property names."""
        yield 'joint_damping'
        yield 'joint_friction'

    def bounds(self):
        """Return an iterator over the bounds for each property."""
        yield self.joint_damping_bounds
        yield self.joint_friction_bounds

    def get_properties(self):
        """
        Get the current physical properties.

        Returns:
            dict: current physical property values.
        """
        joint_dampings = self.joint_dampings
        joint_frictions = self.joint_frictions
        return {joint_id: {'joint_damping': joint_dampings[joint_id], 'joint_friction': joint_frictions[joint_id]}
                for joint_id in self.joints}

    def set_properties(self, properties):
        """
        Set the given physic property values using the simulator.

        Args:
            properties (dict): the physic property values to be set in the simulator.
        """
        # check the given properties
        if not isinstance(properties, dict):
            raise TypeError("Expecting the given 'properties' to be a dictionary, instead got: "
                            "{}".format(type(properties)))

        # set the properties of each joint in the simulator
        if len(properties) > 0:
            for joint_id in self.joints:
                self.simulator.change_dynamics(self.body.id, link_id=joint_id, **properties[joint_id])

    def sample(self, seed=None):
        """
        Sample a new set of physics properties and returns it. Note that it doesn't set them in the simulator.
        This sampling can be useful if the user wishes to check more carefully the sampled physic property values.
        Once satisfied, the user can set them by calling the `set_properties` method.

        Note that it samples uniformly the physics properties between their specified lower and upper bounds.

        Args:
            seed (int, None): random seed.

        Returns:
            dict: sampled physic properties.
        """
        # set random seed
        if seed is not None:
            np.random.seed(seed)

        # sample each property
        properties = dict()
        for i, joint in enumerate(self.joints):
            for name, bound in zip(self.names(), self.bounds()):
                if bound is not None:
                    properties.setdefault(joint, {})[name] = np.random.uniform(low=bound[0][i], high=bound[1][i])
        return properties
