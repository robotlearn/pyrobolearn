#!/usr/bin/env python
"""Define the `LinkPhysicsRandomizer` class which randomizes the physical attributes / properties of a link or
multiple links of a specific body.

Dependencies:
- `pyrobolearn.physics`
"""

import collections

from pyrobolearn.physics.body_physics_randomizer import BodyPhysicsRandomizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinkPhysicsRandomizer(BodyPhysicsRandomizer):
    r"""Link Physics Randomizer

    The link physics randomizer can randomize the physical attributes of a link.
    """

    def __init__(self, body, link_ids=None, masses=None, local_inertia_diagonals=None, local_inertia_positions=None,
                 local_inertia_orientations=None, lateral_frictions=None, spinning_frictions=None,
                 rolling_frictions=None, restitutions=None, linear_dampings=None, angular_dampings=None,
                 contact_stiffnesses=None, contact_dampings=None, **kwargs):
        """
        Initialize the link physics randomizer.

        Args:
            body (Body): multi-body object.
            link_ids (int, list of int, None): link id(s).
            **kwargs (dict): range of possible physical properties. If given one value, that property won't be
                randomized. Each range is a tuple of two values `[lower_bound, upper_bound]`.
        """
        super(LinkPhysicsRandomizer, self).__init__(body)
        self.links = link_ids

        # set the bounds

    ##############
    # Properties #
    ##############

    @property
    def links(self):
        """Return the list of link ids."""
        return self._links

    @links.setter
    def links(self, links):
        """Set the link id or the list of link ids."""
        if isinstance(links, int):
            links = [links]
        elif isinstance(links, collections.Iterable):
            for idx, link in enumerate(links):
                if not isinstance(link, int):
                    raise TypeError("The {} element of the given list of links is not an integer, instead got: "
                                    "{}".format(idx, type(link)))
        else:
            raise TypeError("Expecting the given links to be an integer or a list of integers, instead got: "
                            "{}".format(type(links)))
        self._links = links

    @property
    def masses(self):
        """Return the mass of each specified link."""
        return self.body.get_masses(self.links)

    @masses.setter
    def masses(self, values):
        """Set the mass values."""
        self.body.set_masses(self.links, values)

    @property
    def mass_bounds(self):
        """Return the lower and upper bounds of each link mass."""
        return self._mass_bounds

    @mass_bounds.setter
    def mass_bounds(self, bounds):
        """Set the mass bound for each link."""
        if isinstance(bounds, (float, int)):
            bounds = [(bounds, bounds) for _ in self.links]
        elif isinstance(bounds, (list, tuple, np.ndarray)):
            pass
        self._mass_bounds = bounds

    @property
    def dynamics(self):
        return None

    ###########
    # Methods #
    ###########

    def names(self):
        """Return an iterator over the property names."""
        for name in ['mass']:
            yield name

    def bounds(self):
        """Return an iterator over the bounds for each property."""
        yield self.mass_bounds

    def get_properties(self):
        """
        Get the physics properties.

        Returns:
            dict: current physic property values.
        """
        properties = dict()
        # properties['mass'] =
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

