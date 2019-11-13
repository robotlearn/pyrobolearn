#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the `PhysicsRandomizer` class which randomizes the physical attributes / properties of an object.

This is the main abstract class from which all physics randomizers inherit from.

Dependencies:
- `pyrobolearn.simulators`
"""

import numpy as np

from pyrobolearn.simulators import Simulator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PhysicsRandomizer(object):
    r"""Physics Randomizer

    This the main abstract class from which all the physics randomizers inherit from. A physics randomizer randomize
    the physics properties of a certain object. This can be, for instance, the world, a robot, or a particular link /
    joint of that robot.

    It can change for example the dynamics of a particular object such as the mass, inertia, or others. It can also
    change their physical properties such as the friction, bounciness (restitution coefficient), etc.

    Note that the physics randomizer instance has access to the simulator in order to modify the physical properties.
    Also, note that normally the physics randomizer is called at the beginning of an episode, and not at each time
    step. Changing the physical properties at each time step can lead to weird behaviors.

    It is possible to not randomize some physical properties by specifying a specific value instead of a range (=tuple
    of 2 values; lower and upper bound), or by setting None.
    """

    def __init__(self, simulator):
        """
        Initialize the physics randomizer.

        Args:
            simulator (Simulator): simulator instance
        """
        self.simulator = simulator

    ##############
    # Properties #
    ##############

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self._simulator

    @simulator.setter
    def simulator(self, simulator):
        """Set the simulator instance."""
        if not isinstance(simulator, Simulator):
            raise TypeError("Expecting the given simulator to be an instance of `Simulator`, instead got: "
                            "{}".format(type(simulator)))
        self._simulator = simulator

    ###########
    # Methods #
    ###########

    @staticmethod
    def _check_bounds(name, bounds):
        """Check that the bounds are of the correct type and size.

        Args:
            name (str): name of the physical property
            bounds (list/tuple of float, np.array[2, N], None): bounds. The first item is supposed to be the lower
                bound and the second item the upper bound. If None, it doesn't do anything.
        """
        if bounds is not None:
            # check bounds type and length
            if not isinstance(bounds, (tuple, list, np.ndarray)):
                raise TypeError("Expecting the given '" + name + "' to be a tuple/list/array of len(2) where the "
                                "first item represents the lower bound, and the second item represents the upper "
                                "bound, but got instead a type of: {}".format(type(bounds)))
            if len(bounds) != 2:
                raise ValueError("Expecting the given '" + name + "' to be a tuple/list/array of len(2) where the "
                                 "first item represents the lower bound, and the second item represents the upper "
                                 "bound, but got instead a length of: {}".format(len(bounds)))

    def properties(self):
        """Return an iterator over the properties."""
        properties = self.get_properties()
        for p in properties.values():
            yield p

    def named_properties(self):
        """Return an iterator over the properties with their name and value"""
        properties = self.get_properties()
        for name, p in properties.items():
            yield name, p

    def names(self):
        """Return an iterator over the property names."""
        properties = self.get_properties()
        for name in properties.keys():
            yield name

    def bounds(self):
        """Return an iterator over the bounds for each property."""
        raise NotImplementedError

    def named_bounds(self):
        """Return an iterator over the property bounds with their name and value."""
        for name, bound in zip(self.names(), self.bounds()):
            yield name, bound

    def get_properties(self):
        """
        Get the physics properties. This method has to be implemented in the child class.

        Returns:
            dict: current physic property values {physic property name: corresponding value}
        """
        pass

    def set_properties(self, properties):
        """
        Set the given physic property values using the simulator. This method has to be implemented in the child class.

        Args:
            properties (dict): the physic property values to be set in the simulator.
        """
        pass

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
        for name, bound in zip(self.names(), self.bounds()):
            if bound is not None:
                properties[name] = np.random.uniform(low=bound[0], high=bound[1])
        return properties

    def randomize(self, seed=None):
        """
        Randomize the physics properties and set them in the simulator.

        Args:
            seed (int, None): random seed.
        """
        sampled_properties = self.sample(seed)
        self.set_properties(sampled_properties)

    def seed(self, seed=None):
        """
        Set the given seed when sampling or randomizing the environment.

        Args:
            seed (int): random seed.
        """
        if seed is not None:
            np.random.seed(seed)

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a string describing the class."""
        return

    def __call__(self, seed=None):
        """Randomize the physics properties and set them in the simulator.

        Args:
            seed (int, None): random seed.
        """
        self.randomize(seed=seed)
