#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the `BodyPhysicsRandomizer` class which randomizes the physical attributes / properties of a body.

Dependencies:
- `pyrobolearn.physics`
- `pyrobolearn.robots`
"""

from abc import ABCMeta

from pyrobolearn.physics.physics_randomizer import PhysicsRandomizer
from pyrobolearn.robots.base import Body

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BodyPhysicsRandomizer(PhysicsRandomizer):
    r"""Body Physics Randomizer

    The body physics randomizer can randomize the physical attributes of a body. It is an abstract class which is
    inherited notably by `LinkPhysicsRandomizer` and `JointPhysicsRandomizer`.
    """
    __metaclass__ = ABCMeta

    def __init__(self, body):
        """
        Initialize the body physics randomizer.

        Args:
            body (Body): multi-body object.
        """
        self.body = body
        simulator = self.body.sim
        super(BodyPhysicsRandomizer, self).__init__(simulator)

    ##############
    # Properties #
    ##############

    @property
    def body(self):
        """Return the body / object instance."""
        return self._body

    @body.setter
    def body(self, body):
        """Set the body / object instance."""
        if not isinstance(body, Body):
            raise TypeError("Expecting the given body to be an instance of `Body`, instead got: {}".format(type(body)))
        self._body = body

    @property
    def num_links(self):
        """Return the number of links of the body."""
        return self.body.num_links

    @property
    def num_joints(self):
        """Return the number of joints of the body."""
        return self.body.num_joints
