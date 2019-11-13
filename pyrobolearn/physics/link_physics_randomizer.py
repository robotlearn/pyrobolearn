#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the `LinkPhysicsRandomizer` class which randomizes the physical attributes / properties of a link or
multiple links of a specific body.

Dependencies:
- `pyrobolearn.physics`
"""

import collections
import numpy as np

from pyrobolearn.physics.body_physics_randomizer import BodyPhysicsRandomizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
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
                 contact_stiffnesses=None, contact_dampings=None):
        """
        Initialize the link physics randomizer.

        Args:
            body (Body, Robot): a body or robot instance.
            link_ids (int, list of int, None): link id(s).
            masses (tuple/list of 2 float/np.array[N], np.array[2,N], None): lower and upper bounds for the mass of
                each given link. If None, it won't randomize them.
            local_inertia_diagonals (tuple/list of 2 np.array[3], np.array[2,N,3]): lower and upper bounds for the
                local inertia diagonal for each given link. If None, it won't randomize them.
            local_inertia_positions: lower and upper bounds for the local inertia position for each given link. If
                None, it won't randomize them.
            local_inertia_orientations: lower and upper bounds for the local inertia orientation for each given link.
                If None, it won't randomize them.
            lateral_frictions: lower and upper bounds for the lateral friction for each given link. If None, it won't
                randomize them.
            spinning_frictions: lower and upper bounds for the spinning friction for each given link. If None, it
                won't randomize them.
            rolling_frictions: lower and upper bounds for the rolling friction for each given link. If None, it won't
                randomize them.
            restitutions: lower and upper bounds for the restitution coefficient for each given link. If None, it
                won't randomize them.
            linear_dampings: lower and upper bounds for the linear damping for each given link. If None, it won't
                randomize them.
            angular_dampings: lower and upper bounds for the angular damping for each given link. If None, it won't
                randomize them.
            contact_stiffnesses: lower and upper bounds for the contact stiffness for each given link. If None, it
                won't randomize them.
            contact_dampings: lower and upper bounds for the contact damping for each given link. If None, it won't
                randomize them.
        """
        super(LinkPhysicsRandomizer, self).__init__(body)
        self.links = link_ids

        # set the bounds
        self.mass_bounds = masses
        self.local_inertia_diagonal_bounds = local_inertia_diagonals

        # self.local_inertia_position_bounds = local_inertia_positions
        # self.local_inertia_orientation_bounds = local_inertia_orientations
        # self.linear_damping_bounds = linear_dampings
        # self.angular_damping_bounds = angular_dampings

        self.lateral_friction_bounds = lateral_frictions
        self.spinning_friction_bounds = spinning_frictions
        self.rolling_friction_bounds = rolling_frictions
        self.restitution_bounds = restitutions
        self.contact_stiffness_bounds = contact_stiffnesses
        self.contact_damping_bounds = contact_dampings

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
        if links is None:
            links = list(range(self.body.num_links))
        elif isinstance(links, int):
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
        return [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link)[0] for link in self.links]

    @masses.setter
    def masses(self, values):
        """Set the mass values."""
        # check values
        values = self._check_values('mass', values)

        # set the link masses
        for link, value in zip(self.links, values):
            self.simulator.change_dynamics(body_id=self.body.id, link_id=link, mass=value)

    @property
    def mass_bounds(self):
        """Return the lower and upper bounds of each link's mass."""
        return self._mass_bounds

    @mass_bounds.setter
    def mass_bounds(self, bounds):
        """Set the mass bound for each link."""
        self._check_bounds('masses', bounds)
        self._mass_bounds = bounds

    @property
    def local_inertia_diagonals(self):
        """Return the local inertial diagonal of each specified link."""
        return [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link)[2] for link in self.links]

    @local_inertia_diagonals.setter
    def local_inertia_diagonals(self, values):
        """Set the given local inertia diagonals."""
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given local inertia diagonals to be list/tuple/np.ndarray of float/int, "
                            "but got instead: {}".format(values))
        if len(values) != len(self.links):
            raise ValueError("The number of given mass values (={}) does not match with the number of links "
                             "(={})".format(len(values), len(self.links)))

        # set the link local inertia diagonal
        for link, value in zip(self.links, values):
            if len(value) != 3:
                raise ValueError("Expecting the given local inertia diagonal to be a tuple/list/np.ndarray of length "
                                 "3, but instead got a length of: {}".format(len(value)))
            self.simulator.change_dynamics(body_id=self.body.id, link_id=link, local_inertia_diagonal=value)

    @property
    def local_inertia_diagonal_bounds(self):
        """Return the lower and upper bounds of each link's local inertia diagonal."""
        return self._local_inertia_diagonal_bounds

    @local_inertia_diagonal_bounds.setter
    def local_inertia_diagonal_bounds(self, bounds):
        """Set the local inertia diagonal bound for each link."""
        self._check_bounds('local_inertia_diagonals', bounds)
        self._local_inertia_diagonal_bounds = bounds

    @property
    def local_inertia_positions(self):
        """Return the local inertia position of each specified link."""
        return [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link)[3] for link in self.links]

    @local_inertia_positions.setter
    def local_inertia_positions(self, values):
        """Set the given local inertia positions."""
        raise NotImplementedError("This features is not yet available.")

    @property
    def local_inertia_orientations(self):
        """Return the local inertia orientation of each specified link."""
        return [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link)[4] for link in self.links]

    @local_inertia_orientations.setter
    def local_inertia_orientations(self, values):
        """Set the given local inertia orientations."""
        raise NotImplementedError("This features is not yet available.")

    @property
    def lateral_frictions(self):
        """Return the lateral friction of each specified link."""
        return [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link)[1] for link in self.links]

    @lateral_frictions.setter
    def lateral_frictions(self, values):
        """Set the given lateral frictions."""
        # check values
        self._check_values('lateral_friction', values)

        # set the link masses
        for link, value in zip(self.links, values):
            self.simulator.change_dynamics(body_id=self.body.id, link_id=link, lateral_friction=value)

    @property
    def lateral_friction_bounds(self):
        """Return the lower and upper bounds of each link's lateral friction."""
        return self._lateral_friction_bounds

    @lateral_friction_bounds.setter
    def lateral_friction_bounds(self, bounds):
        """Set the lateral friction bound for each link."""
        self._check_bounds('lateral_frictions', bounds)
        self._lateral_friction_bounds = bounds

    @property
    def restitutions(self):
        """Return the restitution coefficient of each link."""
        return [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link)[5] for link in self.links]

    @restitutions.setter
    def restitutions(self, values):
        """Set the given restitution coefficients."""
        # check values
        self._check_values('restitution', values)

        # set the link masses
        for link, value in zip(self.links, values):
            self.simulator.change_dynamics(body_id=self.body.id, link_id=link, restitution=value)

    @property
    def restitution_bounds(self):
        """Return the lower and upper bounds of each link's restitution coefficient."""
        return self._restitution_bounds

    @restitution_bounds.setter
    def restitution_bounds(self, bounds):
        """Set the restitution coefficient bound for each link."""
        self._check_bounds('restitutions', bounds)
        self._restitution_bounds = bounds

    @property
    def rolling_frictions(self):
        """Return the rolling friction of each link."""
        return [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link)[6] for link in self.links]

    @rolling_frictions.setter
    def rolling_frictions(self, values):
        """Set the given rolling frictions."""
        # check values
        self._check_values('rolling_friction', values)

        # set the link masses
        for link, value in zip(self.links, values):
            self.simulator.change_dynamics(body_id=self.body.id, link_id=link, rolling_friction=value)

    @property
    def rolling_friction_bounds(self):
        """Return the lower and upper bounds of each link's rolling friction."""
        return self._rolling_friction_bounds

    @rolling_friction_bounds.setter
    def rolling_friction_bounds(self, bounds):
        """Set the rolling friction bound for each link."""
        self._check_bounds('rolling_frictions', bounds)
        self._rolling_friction_bounds = bounds

    @property
    def spinning_frictions(self):
        """Return the spinning friction of each link."""
        return [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link)[7] for link in self.links]

    @spinning_frictions.setter
    def spinning_frictions(self, values):
        """Set the given spinning frictions."""
        # check values
        self._check_values('spinning_friction', values)

        # set the link masses
        for link, value in zip(self.links, values):
            self.simulator.change_dynamics(body_id=self.body.id, link_id=link, spinning_friction=value)

    @property
    def spinning_friction_bounds(self):
        """Return the lower and upper bounds of each link's spinning friction."""
        return self._spinning_friction_bounds

    @spinning_friction_bounds.setter
    def spinning_friction_bounds(self, bounds):
        """Set the spinning friction bound for each link."""
        self._check_bounds('spinning_frictions', bounds)
        self._spinning_friction_bounds = bounds

    @property
    def contact_dampings(self):
        """Return the contact damping of each link."""
        return [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link)[8] for link in self.links]

    @contact_dampings.setter
    def contact_dampings(self, values):
        """Set the given contact dampings."""
        # check values
        self._check_values('contact_damping', values)

        # set the link masses
        for link, value in zip(self.links, values):
            self.simulator.change_dynamics(body_id=self.body.id, link_id=link, contact_damping=value)

    @property
    def contact_damping_bounds(self):
        """Return the lower and upper bounds of each link's contact damping."""
        return self._contact_damping_bounds

    @contact_damping_bounds.setter
    def contact_damping_bounds(self, bounds):
        """Set the contact damping bound for each link."""
        self._check_bounds('contact_dampings', bounds)
        self._contact_damping_bounds = bounds

    @property
    def contact_stiffnesses(self):
        """Return the contact stiffness of each link."""
        return [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link)[9] for link in self.links]

    @contact_stiffnesses.setter
    def contact_stiffnesses(self, values):
        """Set the given contact stiffnesses."""
        # check values
        self._check_values('contact_stiffness', values)

        # set the link masses
        for link, value in zip(self.links, values):
            self.simulator.change_dynamics(body_id=self.body.id, link_id=link, contact_stiffness=value)

    @property
    def contact_stiffness_bounds(self):
        """Return the lower and upper bounds of each link's contact stiffness."""
        return self._contact_stiffness_bounds

    @contact_stiffness_bounds.setter
    def contact_stiffness_bounds(self, bounds):
        """Set the contact stiffness bound for each link."""
        self._check_bounds('contact_stiffnesses', bounds)
        self._contact_stiffness_bounds = bounds

    ###########
    # Methods #
    ###########

    def _check_values(self, name, values):
        """Check the given values to be set."""
        if isinstance(values, (float, int)):
            values = [values] * len(self.links)
        elif not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given '" + name + "' to be a list/tuple/np.array of float/int, but got "
                            "instead: {}".format(type(values)))
        if len(values) != len(self.links):
            raise ValueError("The number of given '" + name + "' values (={}) does not match with the number of links "
                             "(={})".format(len(values), len(self.links)))
        return values

    def names(self):
        """Return an iterator over the property names."""
        # 'local_inertia_position', 'local_inertia_orientation', 'linear_damping', 'angular_damping'
        for name in ['mass', 'local_inertia_diagonal', 'lateral_friction', 'spinning_friction', 'rolling_friction',
                     'restitution', 'contact_stiffness', 'contact_damping']:
            yield name

    def bounds(self):
        """Return an iterator over the bounds for each property."""
        yield self.mass_bounds
        yield self.local_inertia_diagonal_bounds
        yield self.lateral_friction_bounds
        yield self.spinning_friction_bounds
        yield self.rolling_friction_bounds
        yield self.restitution_bounds
        yield self.contact_stiffness_bounds
        yield self.contact_damping_bounds

    def get_properties(self):
        """
        Get the current physics properties.

        Returns:
            dict: current physical property values {physic property name: corresponding value}.
        """
        infos = [self.simulator.get_dynamics_info(body_id=self.body.id, link_id=link) for link in self.links]

        return {link: {'mass': info[0], 'local_inertia_diagonal': info[2], 'local_inertia_position': info[3],
                       'local_inertia_orientation': info[4], 'lateral_friction': info[1], 'spinning_friction': info[7],
                       'rolling_friction': info[6], 'restitution': info[5], 'contact_stiffness': info[9],
                       'contact_damping': info[8]}
                for link, info in zip(self.links, infos)}

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

        # set the properties in the simulator
        if len(properties) > 0:
            for link in self.links:
                self.simulator.change_dynamics(self.body.id, link_id=link, **properties[link])

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
        for i, link in enumerate(self.links):
            for name, bound in zip(self.names(), self.bounds()):
                if bound is not None:
                    properties.setdefault(link, {})[name] = np.random.uniform(low=bound[0][i], high=bound[1][i])
        return properties
