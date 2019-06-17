#!/usr/bin/env python
"""Define the `JointPhysicsRandomizer` class which randomizes the physical attributes / properties of a joint or
multiple joints of a specific body.

Dependencies:
- `pyrobolearn.physics`
"""

import collections

from pyrobolearn.physics.body_physics_randomizer import BodyPhysicsRandomizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointPhysicsRandomizer(BodyPhysicsRandomizer):
    r"""Joint Physics Randomizer

    The joint physics randomizer can randomize the physical attributes of a joint. For instance, this can be the
    joint friction or damping coefficients. Other attributes can be the maximum force or velocity the joint(s) can
    achieve.
    """

    def __init__(self, body, joint_ids=None, joint_damping=None, **kwargs):
        """
        Initialize the joint physics randomizer.

        Args:
            body (Body): multi-body object.
            joint_ids (int, list of int, None): joint id(s).
            joint_damping (float, list of float, tuple of float, list of tuple of float, None): joint damping
                coefficient. If None, it will take the default joint damping value associated with the given
                `joint_ids` of the given `body`. If float, it will set that value to the specified joints and will
                always return this value when sampling. If list of float, it will set each value to each joint and will
                always return these values when sampling. If tuple of float, the first item is the lower bound of the
                joint damping and the second item is its upper bound. It will set these bounds for each joint. If list
                of tuples of joints, it will have a tuple of lower / upper bound for each joint.
            **kwargs (dict): range of possible physical properties. If given one value, that property won't be
                randomized. Each range is a tuple of two values `[lower_bound, upper_bound]`.
        """
        super(JointPhysicsRandomizer, self).__init__(body)
        self.joints = joint_ids

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
        elif isinstance(joints, collections.Iterable):
            for idx, joint in enumerate(joints):
                if not isinstance(joint, int):
                    raise TypeError("The {} element of the given list of joints is not an integer, instead got: "
                                    "{}".format(idx, type(joint)))
        else:
            raise TypeError("Expecting the given joints to be an integer or a list of integers, instead got: "
                            "{}".format(type(joints)))
        self._joints = joints

    @property
    def joint_dampings(self):
        """Return the joint dampings associated with the joints."""
        return self.body.get_joint_dampings(self.joints)

    @joint_dampings.setter
    def joint_dampings(self, values):
        """Set the given joint damping values to each joint."""
        for joint, value in zip(self.joints, values):
            self.body.set_joint_damping(joint, value)

    ###########
    # Methods #
    ###########

    def names(self):
        """Return an iterator over the property names."""
        for name in ['joint_damping']:
            yield name

    def bounds(self):
        """Return an iterator over the bounds for each property."""
        pass

    def get_properties(self):
        """
        Get the physics properties.

        Returns:
            dict: current physic property values.
        """
        pass

    def set_properties(self, properties):
        """
        Set the given physic property values using the simulator.

        Args:
            properties (dict): the physic property values to be set in the simulator.
        """
        pass
