#!/usr/bin/env python
"""Define the `WorldPhysicsRandomizer` class which randomizes the physical attributes / properties of the world.

Dependencies:
- `pyrobolearn.physics`
- `pyrobolearn.world`
"""

import numpy as np

from pyrobolearn.worlds import World
from pyrobolearn.physics.physics_randomizer import PhysicsRandomizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WorldPhysicsRandomizer(PhysicsRandomizer):
    r"""World Physics Randomizer

    The world physics randomizer can randomize the physical attributes of a world. It can notably change the gravity,
    the light intensity, the floor friction coefficients (including lateral, spinning, and rolling frictions), etc.
    """

    def __init__(self, world, gravity=None, lateral_friction=None, rolling_friction=None, spinning_friction=None,
                 restitution=None, contact_damping=None, contact_stiffness=None, **kwargs):
        """
        Initialize the world physics randomizer.

        Args:
            world (World): world instance.
            gravity (None, np.float[3], tuple of np.float[3]): gravity bounds. If None, it will take
                the default value returned by the world, and it will not sample from it. If it is a float[3], it will
                set that value to the world and will always return this value when sampling from the physics randomizer.
                If it is a tuple, it has to be of length 2 where the first item is the lower bound and the second item
                is the upper bound.
            lateral_friction (None, float, tuple of float): lateral friction coefficient bounds. If None, it will take
                the default value returned by the world, and it will not sample from it. If it is a float, it will set
                that value to the world and will always return this value when sampling from the physics randomizer.
                If it is a tuple, it has to be of length 2 where the first item is the lower bound and the second item
                is the upper bound.
            rolling_friction (None, float, tuple of float): rolling friction coefficient bounds. (same types as
                described in `lateral_friction`)
            spinning_friction (None, float, tuple of float): spinning friction coefficient bounds. (same types as
                described in `lateral_friction`)
            restitution (None, float, tuple of float): restitution bounds. (same types as described in
                `lateral_friction`)
            contact_damping (None, float, tuple of float): contact damping bounds. (same types as described in
                `lateral_friction`)
            contact_stiffness (None, float, tuple of float): contact stiffness bounds. (same types as described in
                `lateral_friction`)
            **kwargs (dict): range of possible physical properties. If given one value, that property won't be
                randomized. Each range is a tuple of two values `[lower_bound, upper_bound]`.
        """
        self.world = world
        simulator = self.world.simulator
        super(WorldPhysicsRandomizer, self).__init__(simulator)

        # set the bounds
        self.gravity_bounds = gravity
        self.lateral_friction_bounds = lateral_friction
        self.rolling_friction_bounds = rolling_friction
        self.spinning_friction_bounds = spinning_friction
        self.restitution_bounds = restitution
        self.contact_damping_bounds = contact_damping
        self.contact_stiffness_bounds = contact_stiffness

    ##############
    # Properties #
    ##############

    @property
    def world(self):
        """Return the world instance."""
        return self._world

    @world.setter
    def world(self, world):
        """Set the world instance."""
        if not isinstance(world, World):
            raise TypeError("Expecting the given world to be an instance of `World`, instead got: "
                            "{}".format(type(world)))
        self._world = world

    @property
    def gravity(self):
        """Return the gravity vector."""
        return self.world.gravity

    @gravity.setter
    def gravity(self, gravity):
        """
        Set the gravity vector.

        Args:
            (np.float[3]): gravity vector [x,y,z].
        """
        self.world.gravity = gravity

    @property
    def gravity_bounds(self):
        """Return the upper and lower bound for the gravity vector."""
        return self._gravity_bounds

    @gravity_bounds.setter
    def gravity_bounds(self, bounds):
        """Set the upper and lower bounds for the gravity vector."""
        if bounds is None:
            bounds = (self.gravity, self.gravity)
        elif isinstance(bounds, (list, tuple, np.ndarray)):
            if len(bounds) == 2:
                for bound in bounds:
                    if not isinstance(bound, (list, tuple, np.ndarray)):
                        raise TypeError("Expecting one of the bounds to be a 3D vector (list, tuple, or np.ndarray), "
                                        "instead got {}".format(type(bound)))
                    if len(bound) != 3:
                        raise ValueError("Expecting the gravity to be a 3D vector, instead received a {}D "
                                         "vector".format(len(bounds)))
            elif len(bounds) == 3:
                bounds = (bounds, bounds)
            else:
                raise ValueError("Expecting the gravity to be a 3D vector or a tuple of length 2 where the first item "
                                 "is the lower bound and the second item is the upper bound of 3D vectors, instead "
                                 "the given element has a length of {}".format(len(bounds)))
        else:
            raise TypeError("Expecting the gravity bounds to be a 3d vector, a tuple of 3d vectors, or None. Instead "
                            "got {}".format(type(bounds)))
        self._gravity_bounds = bounds

    @property
    def lateral_friction(self):
        """Return the floor lateral friction coefficient."""
        return self.world.lateral_friction

    @lateral_friction.setter
    def lateral_friction(self, coefficient):
        """
        Set the floor lateral friction coefficient.

        Args:
            coefficient (float): lateral friction coefficient.
        """
        self.world.lateral_friction = coefficient

    @property
    def lateral_friction_bounds(self):
        """Return the upper and lower bound for the lateral friction coefficient."""
        return self._lateral_friction_bounds

    @lateral_friction_bounds.setter
    def lateral_friction_bounds(self, bounds):
        """Set the upper and lower bounds for the lateral friction coefficient."""
        if bounds is None:
            bounds = (self.lateral_friction, self.lateral_friction)
        elif isinstance(bounds, float):
            bounds = (bounds, bounds)
        elif isinstance(bounds, (tuple, list)):
            if len(bounds) == 2:
                for bound in bounds:
                    if not isinstance(bound, float):
                        raise TypeError("Expecting one of the bounds to be a float instead got {}".format(type(bound)))
            else:
                raise ValueError("Expecting the bounds to be a tuple of length 2 where the first item is the lower "
                                 "bound and the second item is the upper bound of the lateral friction, instead got "
                                 "a length of {}".format(len(bounds)))
        else:
            raise TypeError("Expecting the lateral friction bounds to be a float, or a tuple of float, or None. "
                            "Instead got {}".format(type(bounds)))
        self._lateral_friction_bounds = bounds

    @property
    def rolling_friction(self):
        """Return the floor rolling friction coefficient."""
        return self.world.rolling_friction

    @rolling_friction.setter
    def rolling_friction(self, coefficient):
        """
        Set the floor rolling friction coefficient.

        Args:
            coefficient (float): rolling friction coefficient.
        """
        self.world.rolling_friction = coefficient

    @property
    def rolling_friction_bounds(self):
        """Return the upper and lower bound for the rolling friction coefficient."""
        return self._rolling_friction_bounds

    @rolling_friction_bounds.setter
    def rolling_friction_bounds(self, bounds):
        """Set the upper and lower bounds for the rolling friction coefficient."""
        if bounds is None:
            bounds = (self.rolling_friction, self.rolling_friction)
        elif isinstance(bounds, float):
            bounds = (bounds, bounds)
        elif isinstance(bounds, (tuple, list)):
            if len(bounds) == 2:
                for bound in bounds:
                    if not isinstance(bound, float):
                        raise TypeError("Expecting one of the bounds to be a float instead got {}".format(type(bound)))
            else:
                raise ValueError("Expecting the bounds to be a tuple of length 2 where the first item is the lower "
                                 "bound and the second item is the upper bound of the rolling friction, instead got "
                                 "a length of {}".format(len(bounds)))
        else:
            raise TypeError("Expecting the rolling friction bounds to be a float, or a tuple of float, or None. "
                            "Instead got {}".format(type(bounds)))
        self._rolling_friction_bounds = bounds

    @property
    def spinning_friction(self):
        """Return the floor spinning friction coefficient."""
        return self.world.spinning_friction

    @spinning_friction.setter
    def spinning_friction(self, coefficient):
        """
        Set the floor spinning friction coefficient.

        Args:
            coefficient (float): spinning friction coefficient.
        """
        self.world.spinning_friction = coefficient

    @property
    def spinning_friction_bounds(self):
        """Return the upper and lower bound for the spinning friction coefficient."""
        return self._spinning_friction_bounds

    @spinning_friction_bounds.setter
    def spinning_friction_bounds(self, bounds):
        """Set the upper and lower bounds for the spinning friction coefficient."""
        if bounds is None:
            bounds = (self.spinning_friction, self.spinning_friction)
        elif isinstance(bounds, float):
            bounds = (bounds, bounds)
        elif isinstance(bounds, (tuple, list)):
            if len(bounds) == 2:
                for bound in bounds:
                    if not isinstance(bound, float):
                        raise TypeError("Expecting one of the bounds to be a float instead got {}".format(type(bound)))
            else:
                raise ValueError("Expecting the bounds to be a tuple of length 2 where the first item is the lower "
                                 "bound and the second item is the upper bound of the spinning friction, instead got "
                                 "a length of {}".format(len(bounds)))
        else:
            raise TypeError("Expecting the spinning friction bounds to be a float, or a tuple of float, or None. "
                            "Instead got {}".format(type(bounds)))
        self._spinning_friction_bounds = bounds

    @property
    def restitution(self):
        """Return the floor restitution (bounciness) coefficient."""
        return self.world.restitution

    @restitution.setter
    def restitution(self, coefficient):
        """
        Set the floor restitution (bounciness) coefficient.

        Args:
            coefficient (float): restitution coefficient.
        """
        self.world.restitution = coefficient

    @property
    def restitution_bounds(self):
        """Return the upper and lower bound for the restitution coefficient."""
        return self._restitution_bounds

    @restitution_bounds.setter
    def restitution_bounds(self, bounds):
        """Set the upper and lower bounds for the restitution coefficient."""
        if bounds is None:
            bounds = (self.restitution, self.restitution)
        elif isinstance(bounds, float):
            bounds = (bounds, bounds)
        elif isinstance(bounds, (tuple, list)):
            if len(bounds) == 2:
                for bound in bounds:
                    if not isinstance(bound, float):
                        raise TypeError("Expecting one of the bounds to be a float instead got {}".format(type(bound)))
            else:
                raise ValueError("Expecting the bounds to be a tuple of length 2 where the first item is the lower "
                                 "bound and the second item is the upper bound of the restitution, instead got "
                                 "a length of {}".format(len(bounds)))
        else:
            raise TypeError("Expecting the restitution bounds to be a float, or a tuple of float, or None. "
                            "Instead got {}".format(type(bounds)))
        self._restitution_bounds = bounds

    @property
    def contact_damping(self):
        """Return the floor contact damping."""
        return self.world.contact_damping

    @contact_damping.setter
    def contact_damping(self, value):
        """
        Set the floor contact damping.

        Args:
            value (float): contact damping value.
        """
        self.world.contact_damping = value

    @property
    def contact_damping_bounds(self):
        """Return the upper and lower bound for the contact damping value."""
        return self._contact_damping_bounds

    @contact_damping_bounds.setter
    def contact_damping_bounds(self, bounds):
        """Set the upper and lower bounds for the contact damping value."""
        if bounds is None:
            bounds = (self.contact_damping, self.contact_damping)
        elif isinstance(bounds, float):
            bounds = (bounds, bounds)
        elif isinstance(bounds, (tuple, list)):
            if len(bounds) == 2:
                for bound in bounds:
                    if not isinstance(bound, float):
                        raise TypeError("Expecting one of the bounds to be a float instead got {}".format(type(bound)))
            else:
                raise ValueError("Expecting the bounds to be a tuple of length 2 where the first item is the lower "
                                 "bound and the second item is the upper bound of the contact damping, instead got "
                                 "a length of {}".format(len(bounds)))
        else:
            raise TypeError("Expecting the contact damping bounds to be a float, or a tuple of float, or None. "
                            "Instead got {}".format(type(bounds)))
        self._contact_damping_bounds = bounds

    @property
    def contact_stiffness(self):
        """Return the floor contact stiffness."""
        return self.world.contact_stiffness

    @contact_stiffness.setter
    def contact_stiffness(self, value):
        """
        Set the floor contact stiffness.

        Args:
            value (float): contact stiffness value.
        """
        self.world.contact_stiffness = value

    @property
    def contact_stiffness_bounds(self):
        """Return the upper and lower bound for the contact stiffness value."""
        return self._contact_stiffness_bounds

    @contact_stiffness_bounds.setter
    def contact_stiffness_bounds(self, bounds):
        """Set the upper and lower bounds for the contact stiffness value."""
        if bounds is None:
            bounds = (self.contact_stiffness, self.contact_stiffness)
        elif isinstance(bounds, float):
            bounds = (bounds, bounds)
        elif isinstance(bounds, (tuple, list)):
            if len(bounds) == 2:
                for bound in bounds:
                    if not isinstance(bound, float):
                        raise TypeError("Expecting one of the bounds to be a float instead got {}".format(type(bound)))
            else:
                raise ValueError("Expecting the bounds to be a tuple of length 2 where the first item is the lower "
                                 "bound and the second item is the upper bound of the contact stiffness, instead got "
                                 "a length of {}".format(len(bounds)))
        else:
            raise TypeError("Expecting the contact stiffness bounds to be a float, or a tuple of float, or None. "
                            "Instead got {}".format(type(bounds)))
        self._contact_stiffness_bounds = bounds

    @property
    def floor_dynamics(self):
        """Return the floor dynamical parameters (friction, restitution, etc).

        Returns:
            float: lateral friction coefficient
            float: rolling friction coefficient
            float: spinning friction coefficient
            float: restitution coefficient
            float: contact damping value
            float: contact stiffness value
        """
        return self.world.floor_dynamics

    @floor_dynamics.setter
    def floor_dynamics(self, dynamics):
        """
        Set the floor dynamics.

        Args:
            values (dict): floor dynamics.
        """
        self.world.floor_dynamics = dynamics

    ###########
    # Methods #
    ###########

    def names(self):
        """Return an iterator over the property names."""
        for name in ['gravity', 'lateral_friction', 'rolling_friction', 'spinning_friction', 'restitution',
                     'contact_damping', 'contact_stiffness']:
            yield name

    def bounds(self):
        """Return an iterator over the bounds"""
        yield self.gravity_bounds
        yield self.lateral_friction_bounds
        yield self.rolling_friction_bounds
        yield self.spinning_friction_bounds
        yield self.restitution_bounds
        yield self.contact_damping_bounds
        yield self.contact_stiffness_bounds

    def get_properties(self):
        """
        Get the physics properties.

        Returns:
            dict: current physic property values.
        """
        properties = dict()
        properties['gravity'] = self.gravity
        floor_dynamics = self.floor_dynamics
        if floor_dynamics is not None:  # there is a floor
            properties['lateral_friction'] = floor_dynamics[0]
            properties['rolling_friction'] = floor_dynamics[1]
            properties['spinning_friction'] = floor_dynamics[2]
            properties['restitution'] = floor_dynamics[3]
            properties['contact_damping'] = floor_dynamics[4]
            properties['contact_stiffness'] = floor_dynamics[5]
        return properties

    def set_properties(self, properties):
        """
        Set the given physic property values using the simulator.

        Args:
            properties (dict): the physic property values to be set in the simulator.
        """
        if not isinstance(properties, dict):
            raise TypeError("Expecting the given 'properties' to be a dictionary, instead got: "
                            "{}".format(type(properties)))

        # set gravity
        if 'gravity' in properties:
            self.gravity = properties['gravity']

        # set floor dynamics
        self.floor_dynamics = properties
