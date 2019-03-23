#!/usr/bin/env python
"""Provide the Legged robot abstract classes.

Classes that are defined here: LeggedRobot, BipedRobot, QuadrupedRobot, HexapodRobot.
"""

import collections
import numpy as np

from pyrobolearn.robots.robot import Robot


class LeggedRobot(Robot):
    r"""Legged robot

    Legged robots are robots that use some end-effectors to move itself. The movement pattern of these end-effectors
    in the standard regime are rhythmic movements.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(LeggedRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling=scaling)

        self.legs = []  # list of legs where a leg is a list of links
        self.feet = []  # list of feet ids

    ##############
    # Properties #
    ##############

    @property
    def num_legs(self):
        """Return the number of legs"""
        return len(self.legs)

    @property
    def num_feet(self):
        """Return the number of feet; this should normally be equal to the number of legs"""
        return len(self.feet)

    ###########
    # Methods #
    ###########

    def get_leg_ids(self, legs=None):
        """
        Return the leg id associated with the given leg index(ices)/name(s).

        Args:
            legs (int, str): leg index(ices) which is [0..num_legs()], or leg name(s)

        Returns:
            int, list[int]: leg id(s)
        """
        if legs is not None:
            if isinstance(legs, int):
                return self.legs[legs]
            elif isinstance(legs, str):
                return self.legs[self.get_link_ids(legs)]
            elif isinstance(legs, (list, tuple)):
                leg_ids = []
                for leg in legs:
                    if isinstance(leg, int):
                        leg_ids.append(self.legs[leg])
                    elif isinstance(leg, str):
                        leg_ids.append(self.legs[self.get_link_ids(leg)])
                    else:
                        raise TypeError("Expecting a str or int for items in legs")
                return leg_ids
        return self.legs

    def get_feet_ids(self, feet=None):
        """
        Return the foot id associated with the given foot index(ices)/name(s).

        Args:
            feet (int, str): foot index(ices) which is [0..num_feet()], or foot name(s)

        Returns:
            int, list[int]: foot id(s)
        """
        if feet is not None:
            if isinstance(feet, int):
                return self.feet[feet]
            elif isinstance(feet, str):
                return self.feet[self.get_link_ids(feet)]
            elif isinstance(feet, (list, tuple)):
                foot_ids = []
                for foot in feet:
                    if isinstance(foot, int):
                        foot_ids.append(self.feet[foot])
                    elif isinstance(foot, str):
                        foot_ids.append(self.feet[self.get_link_ids(foot)])
                    else:
                        raise TypeError("Expecting a str or int for items in feet")
                return foot_ids
        return self.feet

    def set_foot_friction(self, frictions, feet_ids=None):
        """
        Set the foot friction in the simulator.

        Warnings: only available in the simulator.

        Args:
            frictions (float, list of float): friction value(s).
            feet_ids (int, list of int): list of foot/feet id(s).
        """
        if feet_ids is None:
            feet_ids = self.feet
        if isinstance(feet_ids, int):
            feet_ids = [feet_ids]
        if isinstance(frictions, (float, int)):
            frictions = frictions * np.ones(len(feet_ids))
        for foot_id, frict in zip(feet_ids, frictions):
            if isinstance(foot_id, int):
                self.sim.change_dynamics(self.id, foot_id, lateral_friction=frict)
            elif isinstance(foot_id, collections.Iterable):
                for idx in foot_id:
                    self.sim.change_dynamics(self.id, idx, lateral_friction=frict)
            else:
                raise TypeError("Expecting foot_id to be a list of int, or an int. Instead got: "
                                "{}".format(type(foot_id)))

    def center_of_pressure(self):
        """
        Center of Pressure
        """
        # self.sim.getContactPoints(self.id, FootID) # use simulator
        # use F/T sensor to get CoP
        pass

    def zero_moment_point(self):
        """
        Zero Moment Point.
        Assumptions: the contact area is planar and has sufficiently high friction to keep the feet from sliding.
        """
        pass

    def foot_rotation_index(self):
        """
        Foot Rotation Index
        """
        pass

    def divergent_component_motion(self):
        """
        Divergent Component of Motion, a.k.a 'eXtrapolated Center of Mass'
        """
        pass

    def centroidal_moment_pivot(self):
        """
        Centroidal Moment Pivot
        """
        pass

    def draw_support_polygon(self):
        """
        draw the support polygon / convex hull
        """
        pass

    # the following methods need to be overwritten in the children classes

    def move(self, velocity):
        """Move the robot at the specified velocity."""
        pass

    def walk_forward(self):
        """Walk forward."""
        pass

    def walk_backward(self):
        """Walk backward."""
        pass

    def turn_left(self):
        """Turn left."""
        pass

    def turn_right(self):
        """Turn right."""
        pass


class BipedRobot(LeggedRobot):
    r"""Biped Robot

    A biped robot is a robot which has 2 legs.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(BipedRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)

        self.left_leg_id = 0
        self.right_leg_id = 1

    ##############
    # Properties #
    ##############

    @property
    def left_leg(self):
        """Return the left leg joint ids"""
        return self.legs[self.left_leg_id]

    @property
    def right_leg(self):
        """Return the right leg joint ids"""
        return self.legs[self.right_leg_id]

    @property
    def left_foot(self):
        """Return the left foot id"""
        return self.feet[self.left_leg_id]

    @property
    def right_foot(self):
        """Return the right foot id"""
        return self.feet[self.right_leg_id]


class QuadrupedRobot(LeggedRobot):
    r"""Quadruped robot

    A quadruped robot is a robot which has 4 legs.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(QuadrupedRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)

        self.left_front_leg_id = 0
        self.right_front_leg_id = 1
        self.left_back_leg_id = 2
        self.right_back_leg_id = 3

    ##############
    # Properties #
    ##############

    @property
    def left_front_leg(self):
        """Return the left front leg joint ids"""
        return self.legs[self.left_front_leg_id]

    @property
    def right_front_leg(self):
        """Return the right front leg joint ids"""
        return self.legs[self.right_front_leg_id]

    @property
    def left_back_leg(self):
        """Return the left back leg joint ids"""
        return self.legs[self.left_back_leg_id]

    @property
    def right_back_leg(self):
        """Return the right back leg joint ids"""
        return self.legs[self.right_back_leg_id]

    @property
    def left_front_foot(self):
        """Return the left front foot id"""
        return self.feet[self.left_front_leg_id]

    @property
    def right_front_foot(self):
        """Return the right front foot id"""
        return self.feet[self.right_front_leg_id]

    @property
    def left_back_foot(self):
        """Return the left back foot id"""
        return self.feet[self.left_back_leg_id]

    @property
    def right_back_foot(self):
        """Return the right back foot id"""
        return self.feet[self.right_back_leg_id]


class HexapodRobot(LeggedRobot):
    r"""Hexapod Robot

    An hexapod robot is a robot which has 6 legs.
    """

    def __init__(self, simulator, urdf, position, orientation=None, fixed_base=False, scaling=1.):
        super(HexapodRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)

        self.left_front_leg_id = 0
        self.right_front_leg_id = 1
        self.left_middle_leg_id = 2
        self.right_middle_leg_id = 3
        self.left_back_leg_id = 4
        self.right_back_leg_id = 5

    ##############
    # Properties #
    ##############

    @property
    def left_front_leg(self):
        """Return the left front leg ids"""
        return self.legs[self.left_front_leg_id]

    @property
    def right_front_leg(self):
        """Return the right front leg ids"""
        return self.legs[self.right_front_leg_id]

    @property
    def left_middle_leg(self):
        """Return the left middle leg ids"""
        return self.legs[self.left_middle_leg_id]

    @property
    def right_middle_leg(self):
        """Return the right middle leg ids"""
        return self.legs[self.right_middle_leg_id]

    @property
    def left_back_leg(self):
        """Return the left back leg ids"""
        return self.legs[self.left_back_leg_id]

    @property
    def right_back_leg(self):
        """Return the right back leg ids"""
        return self.legs[self.right_back_leg_id]

    @property
    def left_front_foot(self):
        """Return the left front foot id"""
        return self.feet[self.left_front_leg_id]

    @property
    def right_front_foot(self):
        """Return the right front foot id"""
        return self.feet[self.right_front_leg_id]

    @property
    def left_middle_foot(self):
        """Return the left middle foot id"""
        return self.feet[self.left_middle_leg_id]

    @property
    def right_middle_foot(self):
        """Return the right middle foot id"""
        return self.feet[self.right_middle_leg_id]

    @property
    def left_back_foot(self):
        """Return the left back foot id"""
        return self.feet[self.left_back_leg_id]

    @property
    def right_back_foot(self):
        """Return the right back foot id"""
        return self.feet[self.right_back_leg_id]
