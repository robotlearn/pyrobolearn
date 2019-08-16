#!/usr/bin/env python
"""Define the costs used on link states / actions.
"""

from abc import ABCMeta
import numpy as np

import pyrobolearn as prl
from pyrobolearn.rewards.reward import Reward


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GeometricalReward(Reward):
    r"""Geometrical reward.

    The geometrical reward uses 3D geometric shapes to describe a reward function. This can notably be used for reward
    shaping to guide the agent in the 3D space.
    """

    def __init__(self):
        super(GeometricalReward, self).__init__()

        self._visual = None

    def _compute(self):
        """Compute and return the reward value."""
        pass

    def draw(self):
        """Draw the visual shape in the simulator."""
        pass


class SphericalReward(GeometricalReward):
    r"""Spherical reward.

    """

    def __init__(self, bodies, link_ids=-1, attached_body=None, attached_link_id=-1, position=None, orientation=None,
                 radius=1, theta=(0, 2*np.pi), phi=(0, np.pi), radius_reward_range=None, theta_reward_ranges=None,
                 height_reward_range=None, interpolation='linear', simulator=None):
        """
        Initialize the spherical reward.

        Args:
            bodies (int, Body, list[int], list[Body]): the bodies or body unique ids that we should check if they are
                inside the sphere.
            link_ids (int, list[int]): the link id associated to each body. By default, it is -1 for the base.
            attached_body (Body, int, None): the body instance or unique id to which the spherical reward is
                attached to. If None, it will be attached to the world frame.
            attached_link_id (int, None): the link id to which the reward is attached to. By default, it is -1 for the
                base.
            position (np.array/list/tuple[float[3]], None): local position of the spherical reward. If None, it will
                be the origin (0,0,0).
            orientation (np.array/list/tuple[float[4]], None): local orientation (expressed as a quaternion [x,y,z,w])
                of the spherical reward. If None, it will be the unit quaternion [0,0,0,1].
            radius (float, tuple[float[2]]): radius of the sphere. If two radii are provided, the first one is the
                inner radius and the second one is the outer radius of the sphere.
            theta (tuple[float[2]]): the lower and upper bounds of the theta angle.
            phi (tuple[float[2]]): the lower and upper bounds of the phi angle.
            radius_reward_range (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(r,v)` where the first item represents the radius value
                `r` and the second item represents the associated reward value `v`.
            theta_reward_ranges (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(t,v)` where the first item represents the theta angle
                value `t` and the second item represents the associated reward value `v`.
            height_reward_range (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(h,v)` where the first item represents the height value
                `h` and the second item represents the associated reward value `v`.
            interpolation (str): the interpolation method to use for the given reward ranges. Currently, you can select
                between 'linear' or 'step'.
            simulator (Simulator, None): if the given bodies are all unique ids, the simulator instance has to be
                provided.
        """
        pass

    def _compute(self):
        """Compute and return the reward value."""
        pass

    def draw(self):
        """Draw the visual shape in the simulator."""
        # if self._visual is None:
        #     visual_shape = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)
        #     sphere = self.sim.create_body(visual_shape_id=visual_shape, mass=0., position=position)
        pass


class RectangularReward(GeometricalReward):
    r"""Rectangular reward

    """

    def __init__(self, bodies, link_ids=-1, attached_body=None, attached_link_id=-1, position=None, orientation=None,
                 a=1, b=1, c=1, radius_reward_range=None, theta_reward_ranges=None,
                 height_reward_range=None, interpolation='linear', simulator=None):
        """
        Initialize the rectangular reward.

        Args:
            bodies (int, Body, list[int], list[Body]): the bodies or body unique ids that we should check if they are
                inside the rectangle.
            link_ids (int, list[int]): the link id associated to each body. By default, it is -1 for the base.
            attached_body (Body, int, None): the body instance or unique id to which the rectangular reward is
                attached to. If None, it will be attached to the world frame.
            attached_link_id (int, None): the link id to which the reward is attached to. By default, it is -1 for the
                base.
            position (np.array/list/tuple[float[3]], None): local position of the rectangular reward. If None, it will
                be the origin (0,0,0).
            orientation (np.array/list/tuple[float[4]], None): local orientation (expressed as a quaternion [x,y,z,w])
                of the rectangular reward. If None, it will be the unit quaternion [0,0,0,1].
            a (float, tuple[float[2]]): radius of the rectangle. If two radii are provided, the first one is the
                inner radius and the second one is the outer radius of the rectangle.
            b (tuple[float[2]]): the lower and upper bound of the
            c (float): the height/length of the rectangle.
            radius_reward_range (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(r,v)` where the first item represents the radius value
                `r` and the second item represents the associated reward value `v`.
            theta_reward_ranges (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(t,v)` where the first item represents the theta angle
                value `t` and the second item represents the associated reward value `v`.
            height_reward_range (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(h,v)` where the first item represents the height value
                `h` and the second item represents the associated reward value `v`.
            interpolation (str): the interpolation method to use for the given reward ranges. Currently, you can select
                between 'linear' or 'step'.
            simulator (Simulator, None): if the given bodies are all unique ids, the simulator instance has to be
                provided.
        """
        pass

    def _compute(self):
        """Compute and return the reward value."""
        pass

    def draw(self):
        """Draw the visual shape in the simulator."""
        pass


class EllipsoidalReward(GeometricalReward):
    r"""Ellipsoidal reward

    """

    def __init__(self, bodies, link_ids=-1, attached_body=None, attached_link_id=-1, position=None, orientation=None,
                 a=1, b=1, c=1, radius_reward_range=None, theta_reward_ranges=None,
                 height_reward_range=None, interpolation='linear', simulator=None):
        """
        Initialize the ellipsoidal reward.

        Args:
            bodies (int, Body, list[int], list[Body]): the bodies or body unique ids that we should check if they are
                inside the ellipsoid.
            link_ids (int, list[int]): the link id associated to each body. By default, it is -1 for the base.
            attached_body (Body, int, None): the body instance or unique id to which the ellipsoidal reward is
                attached to. If None, it will be attached to the world frame.
            attached_link_id (int, None): the link id to which the reward is attached to. By default, it is -1 for the
                base.
            position (np.array/list/tuple[float[3]], None): local position of the ellipsoidal reward. If None, it will
                be the origin (0,0,0).
            orientation (np.array/list/tuple[float[4]], None): local orientation (expressed as a quaternion [x,y,z,w])
                of the ellipsoidal reward. If None, it will be the unit quaternion [0,0,0,1].
            a (float, tuple[float[2]]): length of the first semi-axis. If tuple, it is the lower and upper bounds of
                the length of the first semi-axis.
            b (float, tuple[float[2]]): length of the second semi-axis. If tuple, it is the lower and upper bounds of
                the length of the second semi-axis.
            c (float, tuple[float[2]]): length of the third semi-axis. If tuple, it is the lower and upper bounds of
                the length of the third semi-axis.
            radius_reward_range (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(r,v)` where the first item represents the radius value
                `r` and the second item represents the associated reward value `v`.
            theta_reward_ranges (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(t,v)` where the first item represents the theta angle
                value `t` and the second item represents the associated reward value `v`.
            height_reward_range (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(h,v)` where the first item represents the height value
                `h` and the second item represents the associated reward value `v`.
            interpolation (str): the interpolation method to use for the given reward ranges. Currently, you can select
                between 'linear' or 'step'.
            simulator (Simulator, None): if the given bodies are all unique ids, the simulator instance has to be
                provided.
        """
        pass

    def _compute(self):
        """Compute and return the reward value."""
        pass

    def draw(self):
        """Draw the visual shape in the simulator."""
        pass


class CylindricalReward(GeometricalReward):
    r"""Cylindrical reward
    """

    def __init__(self, bodies, link_ids=-1, attached_body=None, attached_link_id=-1, position=None, orientation=None,
                 radius=1, theta=(0, 2*np.pi), height=1, radius_reward_range=None, theta_reward_ranges=None,
                 height_reward_range=None, interpolation='linear', simulator=None):
        """
        Initialize the cylindrical reward.

        Args:
            bodies (int, Body, list[int], list[Body]): the bodies or body unique ids that we should check if they are
                inside the cylinder.
            link_ids (int, list[int]): the link id associated to each body. By default, it is -1 for the base.
            attached_body (Body, int, None): the body instance or unique id to which the cylindrical reward is
                attached to. If None, it will be attached to the world frame.
            attached_link_id (int, None): the link id to which the reward is attached to. By default, it is -1 for the
                base.
            position (np.array/list/tuple[float[3]], None): local position of the cylindrical reward. If None, it will
                be the origin (0,0,0).
            orientation (np.array/list/tuple[float[4]], None): local orientation (expressed as a quaternion [x,y,z,w])
                of the cylindrical reward. If None, it will be the unit quaternion [0,0,0,1].
            radius (float, tuple[float[2]]): radius of the cylinder. If two radii are provided, the first one is the
                inner radius and the second one is the outer radius of the cylinder.
            theta (tuple[float[2]]): the lower and upper bounds of the theta angle.
            height (float): the height/length of the cylinder.
            radius_reward_range (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(r,v)` where the first item represents the radius value
                `r` and the second item represents the associated reward value `v`.
            theta_reward_ranges (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(t,v)` where the first item represents the theta angle
                value `t` and the second item represents the associated reward value `v`.
            height_reward_range (list[tuple[float[2]]], list/tuple[float[2]]): If a list / tuple of 2 floats, it is
                the lower and upper bounds of the reward range. If a list of tuple of 2 floats, for each item
                in the list, it must be a tuple of length 2 `(h,v)` where the first item represents the height value
                `h` and the second item represents the associated reward value `v`.
            interpolation (str): the interpolation method to use for the given reward ranges. Currently, you can select
                between 'linear' or 'step'.
            simulator (Simulator, None): if the given bodies are all unique ids, the simulator instance has to be
                provided.
        """
        pass

    def _compute(self):
        """Compute and return the reward value."""
        pass

    def draw(self):
        """Draw the visual shape in the simulator."""
        pass


class CompositeGeometricalReward(GeometricalReward):
    r"""Composite geometrical reward

    This class is useful if you have overlapping geometrical rewards and you wish to prioritize some shapes over others
    when computing the reward function.
    """

    def __init__(self, rewards, priorities):
        r"""
        Initialize the composite geometrical reward.

        Args:
            rewards (list[GeometricalReward]): list of geometrical rewards.
            priorities (list[int]): list of priorities, where each priority is associated with the given geometrical
                reward. The length of this list must match the number of given rewards. Two rewards can have the same
                priorities, and if they overlap their returned reward values are added.
        """
        super(CompositeGeometricalReward, self).__init__()
        self.rewards = rewards
        self.priorities = priorities

    def _compute(self):
        """Compute and return the reward value."""
        pass

    def draw(self):
        """Draw the visual shape in the simulator."""
        pass
