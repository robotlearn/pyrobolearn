# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define world actions

This includes:

- AttachAction: this allows you to attach / detach a link with another link.
"""

from abc import ABCMeta
import copy
import numpy as np

from pyrobolearn.actions.action import Action
from pyrobolearn.robots.base import Body
from pyrobolearn.worlds.world import World


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WorldAction(Action):
    r"""World action (abstract)

    This provides the abstract class that allows to perform an action in the world. This includes to attach or detach
    two bodies, and others.
    """
    __metaclass__ = ABCMeta

    def __init__(self, world):
        """
        Initialize the world action.

        Args:
            world (World): world instance.
        """
        super(WorldAction, self).__init__()
        self.world = world

    @property
    def world(self):
        """Return the world instance."""
        return self._world

    @world.setter
    def world(self, world):
        """Set the world instance."""
        self._world = world

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(world=self.world)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        world = copy.deepcopy(self.world)
        action = self.__class__(world=world)
        memo[self] = action
        return action


class AttachAction(WorldAction):
    r"""Attach Action.

    The attach action is a discrete action which can take two values: 0 (=detach) or 1 (=attach). This allows to
    attach a robot's link with another body's link in the world. Note that is only valid in the simulator. In order to
    attach the robot's link with the other link, they both have to be close to each other.

    Warnings:
        - This is only valid in the simulator.
        - Currently, the other link id to which we would like to attach has to be provided.
    """

    def __init__(self, world, body1, body2, link_id1=-1, link_id2=-1, distance_threshold=0.1,
                 body1_frame_position=(0., 0., 0.), body2_frame_position=(0., 0., 0.),
                 body1_frame_orientation=None, body2_frame_orientation=None):
        """
        Initialize the attach action.

        Args:
            world (World): world instance.
            body1 (Body): first body instance.
            body2 (Body): second body instance.
            link_id1 (int): unique link id of the first body instance.
            link_id2 (int): unique link id of the second body instance.
            distance_threshold (float): distance threshold between the two links such that they can be attached.
            body1_frame_position (np.array[3]): position of the joint frame relative to parent CoM frame.
            body2_frame_position (np.array[3]): position of the joint frame relative to a given child CoM frame (or
                world origin if no child specified)
            body1_frame_orientation (np.array[4]): the orientation of the joint frame relative to parent CoM
                coordinate frame (expressed as a quaternion [x,y,z,w])
            body2_frame_orientation (np.array[4]): the orientation of the joint frame relative to the child CoM
                coordinate frame, or world origin frame if no child specified (expressed as a quaternion [x,y,z,w])
        """
        super(AttachAction, self).__init__(world=world)

        # check body instances
        def check_body(body, name):
            if not isinstance(body, Body):
                raise TypeError("Expecting the given '" + name + "' to be an instance of `Body`, but got instead: "
                                "{}".format(type(body)))
            return body

        self._body1 = check_body(body1, 'body1')
        self._body2 = check_body(body2, 'body2')

        # check links
        def check_link(link, name):
            if link is None:
                link = -1
            if not isinstance(link, int):
                raise TypeError("Expecting the given '" + name + "' to be an int, but got instead: "
                                                                 "{}".format(type(link)))
            return link

        self._link1 = check_link(link_id1, 'link_id1')
        self._link2 = check_link(link_id2, 'link_id2')

        # check distance threshold
        if not isinstance(distance_threshold, (float, int)):
            raise TypeError("Expecting the given 'distance_threshold' to be a float or int, but got instead: "
                            "{}".format(type(distance_threshold)))
        if distance_threshold < 0:
            raise ValueError("The given 'distance_threshold' should be a positive number.")
        self._distance_threshold = distance_threshold

        # set other variables
        self._body1_frame_position = body1_frame_position
        self._body2_frame_position = body2_frame_position
        self._body1_frame_orientation = body1_frame_orientation
        self._body2_frame_orientation = body2_frame_orientation

        # variable to remember if they are already attached or not
        self._attached = False

    def _write(self, data):
        """
        Write the data.

        Args:
            data (int, np.ndarray): the binary data; 0 = detach and 1 = attach.
        """
        # get data
        if isinstance(data, np.ndarray):
            data = data[0]

        if data == 1:  # attach
            if not self._attached:  # if not already attached

                # check distance
                results = self.world.get_closest_bodies(body=self._body1, radius=self._distance_threshold,
                                                        link_id=self._link1, body2=self._body2, link2_id=self._link2)

                # if found body2 in close vicinity of body1
                if len(results) > 0:
                    self.world.attach(body1=self._body1, body2=self._body2, link1=self._link1, link2=self._link2,
                                      parent_frame_position=self._body1_frame_position,
                                      child_frame_position=self._body2_frame_position,
                                      parent_frame_orientation=self._body1_frame_orientation,
                                      child_frame_orientation=self._body2_frame_orientation)
                    self._attached = not self._attached

        else:  # detach
            if self._attached:  # if attached
                self.world.detach(body1=self._body1, body2=self._body2, link1=self._link1, link2=self._link2)
                self._attached = not self._attached

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(world=self.world, body1=self._body1, body2=self._body2, link_id1=self._link1,
                              link_id2=self._link2, distance_threshold=self._distance_threshold,
                              body1_frame_position=self._body1_frame_position,
                              body2_frame_position=self._body2_frame_position,
                              body1_frame_orientation=self._body1_frame_orientation,
                              body2_frame_orientation=self._body2_frame_orientation)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        world = copy.deepcopy(self.world)
        body1 = copy.deepcopy(self._body1)
        body2 = copy.deepcopy(self._body2)
        body1_frame_position = copy.deepcopy(self._body1_frame_position)
        body2_frame_position = copy.deepcopy(self._body2_frame_position)
        body1_frame_orientation = copy.deepcopy(self._body1_frame_orientation)
        body2_frame_orientation = copy.deepcopy(self._body2_frame_orientation)
        action = self.__class__(world=world, body1=body1, body2=body2, link_id1=self._link1, link_id2=self._link2,
                                body1_frame_position=body1_frame_position, body2_frame_position=body2_frame_position,
                                body1_frame_orientation=body1_frame_orientation,
                                body2_frame_orientation=body2_frame_orientation)
        memo[self] = action
        return action
