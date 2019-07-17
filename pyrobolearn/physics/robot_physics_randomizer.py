#!/usr/bin/env python
"""Define the `RobotPhysicsRandomizer` class which randomizes the physical attributes / properties of links and joints.

Dependencies:
- `pyrobolearn.physics`
"""

import collections

from pyrobolearn.physics.body_physics_randomizer import BodyPhysicsRandomizer
from pyrobolearn.physics.link_physics_randomizer import LinkPhysicsRandomizer
from pyrobolearn.physics.joint_physics_randomizer import JointPhysicsRandomizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotPhysicsRandomizer(BodyPhysicsRandomizer):
    r"""Robot Physics Randomizer

    The robot physics randomizer can randomize the physical attributes of a robot. It can notably change its mass,
    the contact frictions, the inertia of the links, the friction and damping coefficients of the joints, etc.
    """

    def __init__(self, body, links=None, joints=None, **kwargs):
        """
        Initialize the robot physics randomizer.

        Args:
            body (Body): multi-body object.
            links (int, list of int, LinkPhysicsRandomizer, list of LinkPhysicsRandomizer, None): link id(s) or link
                physics randomizer(s).
            joints (int, list of int, JointPhysicsRandomizer, list JointPhysicsRandomizer, None): joint id(s) or joint
                physics randomizer.
            **kwargs (dict): range of possible physical properties. If given one value, that property won't be
                randomized. Each range is a tuple of two values `[lower_bound, upper_bound]`.
        """
        super(RobotPhysicsRandomizer, self).__init__(body)
        self.links = links
        self.joints = joints

    ##############
    # Properties #
    ##############

    @property
    def links(self):
        """Return the list of link physics randomizers."""
        return self._links

    @links.setter
    def links(self, links):
        """Set the link physics randomizer or the list of link physics randomizers."""
        if isinstance(links, int):
            links = [LinkPhysicsRandomizer(self.body, links)]
        elif isinstance(links, LinkPhysicsRandomizer):
            links = [links]
        elif isinstance(links, collections.Iterable):
            link_list = []
            for idx, link in enumerate(links):
                if isinstance(link, int):
                    link = LinkPhysicsRandomizer(self.body, links)
                elif not isinstance(link, LinkPhysicsRandomizer):
                    raise TypeError("The {} element of the given list of links is not an integer or a "
                                    "LinkPhysicsRandomizer, instead got: {}".format(idx, type(link)))
                link_list.append(link)
            links = link_list
        else:
            raise TypeError("Expecting the given links to be an integer / `LinkPhysicsRandomizer` or a list of "
                            "integers / `LinkPhysicsRandomizer`, instead got: {}".format(type(links)))
        self._links = links

    @property
    def joints(self):
        """Return the list of joint physics randomizers."""
        return self._joints

    @joints.setter
    def joints(self, joints):
        """Set the joint physics randomizer or the list of joint physics randomizers."""
        if isinstance(joints, int):
            joints = [JointPhysicsRandomizer(self.body, joints)]
        elif isinstance(joints, JointPhysicsRandomizer):
            joints = [joints]
        elif isinstance(joints, collections.Iterable):
            joint_list = []
            for idx, joint in enumerate(joints):
                if isinstance(joint, int):
                    joint = JointPhysicsRandomizer(self.body, joints)
                elif not isinstance(joint, JointPhysicsRandomizer):
                    raise TypeError("The {} element of the given list of joints is not an integer or a "
                                    "JointPhysicsRandomizer, instead got: {}".format(idx, type(joint)))
                joint_list.append(joint)
            joints = joint_list
        else:
            raise TypeError("Expecting the given joints to be an integer / `JointPhysicsRandomizer` or a list of "
                            "integers / `JointPhysicsRandomizer`, instead got: {}".format(type(joints)))
        self._joints = joints

    ###########
    # Methods #
    ###########

    def names(self):
        """Return an iterator over the property names."""
        pass

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
