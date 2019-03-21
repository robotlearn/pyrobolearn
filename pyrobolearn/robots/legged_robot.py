#!/usr/bin/env python
"""Provide the Legged robot abstract classes.
"""

import collections
import numpy as np
from robot import Robot


class LeggedRobot(Robot):
    r"""Legged robot

    Legged robots are robots that use some end-effectors to move itself. The movement pattern of these end-effectors
    in the standard regime are rhythmic movements.
    """

    def __init__(self,
                 simulator,
                 urdf_path,
                 init_pos=(0, 0, 1.),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.):
        super(LeggedRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling=1.)

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

    def getNumberOfLegs(self):
        """
        Return the number of legs/feet.

        Returns:
            int: the number of legs/feet
        """
        return self.num_legs

    def getLegLinkIds(self, legIds=None):
        pass

    def getLegLinkNames(self, legIds=None):
        pass

    def getFeetIds(self, footId=None):
        pass

    def getFeetNames(self, footIds=None):
        pass

    def getCoP(self):
        """
        Center of Pressure

        Returns:

        """
        # self.sim.getContactPoints(self.id, FootID) # use simulator
        # use F/T sensor to get CoP
        pass

    def getZMP(self):
        """
        Zero Moment Point.
        Assumptions: the contact area is planar and has sufficiently high friction to keep the feet from sliding.

        Returns:

        """
        pass

    def getFRI(self):
        """
        Foot Rotation Index

        Returns:

        """
        pass

    def getDCM(self):
        """
        Divergent Component of Motion, a.k.a 'eXtrapolated Center of Mass'

        Returns:

        """
        pass

    def getCMP(self):
        """
        Centroidal Moment Pivot

        Returns:

        """

    def drawSupportPolygon(self):
        """
        draw the support polygon / convex hull
        """
        pass

    # the following methods need to be overwritten in the children classes

    def move(self, velocity):
        raise NotImplementedError

    def walkForward(self):
        raise NotImplementedError

    def walkBackward(self):
        raise NotImplementedError

    def turnLeft(self):
        raise NotImplementedError

    def turnRight(self):
        raise NotImplementedError

    def setFootFriction(self, friction, feet_id=None):
        """
        Set the foot friction in the simulator.

        Warnings: only available in the simulator.

        Args:
            friction (float, list of float): friction value(s).
            feet_id (int, list of int): list of foot/feet id(s).
        """
        if feet_id is None:
            foot_id = self.feet
        if isinstance(feet_id, int):
            feet_id = [feet_id]
        if isinstance(friction, (float, int)):
            friction = friction * np.ones(len(feet_id))
        for foot_id, frict in zip(feet_id, friction):
            if isinstance(foot_id, int):
                self.sim.changeDynamics(self.id, foot_id, lateralFriction=frict)
            elif isinstance(foot_id, collections.Iterable):
                for idx in foot_id:
                    self.sim.changeDynamics(self.id, idx, lateralFriction=frict)
            else:
                raise TypeError("Expecting foot_id to be a list of int, or an int. Instead got: "
                                "{}".format(type(foot_id)))


class BipedRobot(LeggedRobot):
    r"""Biped Robot

    """

    def __init__(self, simulator, urdf_path, init_pos=(0,0,1.5), init_orient=(0,0,0,1), useFixedBase=False, scaling=1.):
        super(BipedRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)

        self.left_leg_id = 0
        self.right_leg_id = 1

    ##############
    # Properties #
    ##############

    @property
    def left_leg(self):
        return self.legs[self.left_leg_id]

    @property
    def right_leg(self):
        return self.legs[self.right_leg_id]

    ###########
    # Methods #
    ###########

    def getLeftLegIds(self):
        """
        Return the left leg actuated joint/link ids.
        """
        return self.legs[self.left_leg_id]

    def getRightLegIds(self):
        """
        Return the right leg actuated joint/link ids.
        """
        return self.legs[self.right_leg_id]

    def getLeftFootId(self):
        """
        Return the left foot id.
        """
        return self.feet[self.left_leg_id]

    def getRightFootId(self):
        """
        Return the right foot id.
        """
        return self.feet[self.right_leg_id]


class QuadrupedRobot(LeggedRobot):
    r"""Quadruped robot

    """

    def __init__(self, simulator, urdf_path, init_pos=(0,0,1.), init_orient=(0,0,0,1), useFixedBase=False, scaling=1.):
        super(QuadrupedRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)

        self.left_front_leg_id = 0
        self.right_front_leg_id = 1
        self.left_back_leg_id = 2
        self.right_back_leg_id = 3

    ##############
    # Properties #
    ##############

    @property
    def left_front_leg(self):
        return self.legs[self.left_front_leg_id]

    @property
    def right_front_leg(self):
        return self.legs[self.right_front_leg_id]

    @property
    def left_back_leg(self):
        return self.legs[self.left_back_leg_id]

    @property
    def right_back_leg(self):
        return self.legs[self.right_back_leg_id]

    ###########
    # Methods #
    ###########

    def getLeftFrontLegIds(self):
        """Return the left front leg ids"""
        return self.legs[self.left_front_leg_id]

    def getLeftFrontFootId(self):
        """Return the left front foot id"""
        return self.feet[self.left_front_leg_id]

    def getRightFrontLegIds(self):
        """Return the right front leg ids"""
        return self.legs[self.right_front_leg_id]

    def getRightFrontFootId(self):
        """Return the right front foot id"""
        return self.feet[self.right_front_leg_id]

    def getLeftBackLegIds(self):
        """Return the left back leg ids"""
        return self.legs[self.left_back_leg_id]

    def getLeftBackFootId(self):
        """Return the left back foot id"""
        return self.feet[self.left_back_leg_id]

    def getRightBackLegIds(self):
        """Return the right back leg ids"""
        return self.legs[self.right_back_leg_id]

    def getRightBackFootId(self):
        """Return the right back foot id"""
        return self.feet[self.right_back_leg_id]


class HexapodRobot(LeggedRobot):

    def __init__(self, simulator, urdf_path, init_pos=(0,0,1.), init_orient=(0,0,0,1), useFixedBase=False, scaling=1.):
        super(HexapodRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)

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
        return self.legs[self.left_front_leg_id]

    @property
    def right_front_leg(self):
        return self.legs[self.right_front_leg_id]

    @property
    def left_middle_leg(self):
        return self.legs[self.left_middle_leg_id]

    @property
    def right_middle_leg(self):
        return self.legs[self.right_middle_leg_id]

    @property
    def left_back_leg(self):
        return self.legs[self.left_back_leg_id]

    @property
    def right_back_leg(self):
        return self.legs[self.right_back_leg_id]

    ###########
    # Methods #
    ###########

    def getLeftFrontLegIds(self):
        """Return the left front leg ids"""
        return self.legs[self.left_front_leg_id]

    def getLeftFrontFootId(self):
        """Return the left front foot id"""
        return self.feet[self.left_front_leg_id]

    def getRightFrontLegIds(self):
        """Return the right front leg ids"""
        return self.legs[self.right_front_leg_id]

    def getRightFrontFootId(self):
        """Return the right front foot id"""
        return self.feet[self.right_front_leg_id]

    def getLeftMiddleLegIds(self):
        """Return the left middle leg ids"""
        return self.legs[self.left_middle_leg_id]

    def getLeftMiddleFootId(self):
        """Return the left middle foot id"""
        return self.feet[self.left_middle_leg_id]

    def getRightMiddleLegIds(self):
        """Return the right middle leg ids"""
        return self.legs[self.right_middle_leg_id]

    def getRightMiddleFootId(self):
        """Return the right middle foot id"""
        return self.feet[self.right_middle_leg_id]

    def getLeftBackLegIds(self):
        """Return the left back leg ids"""
        return self.legs[self.left_back_leg_id]

    def getLeftBackFootId(self):
        """Return the left back foot id"""
        return self.feet[self.left_back_leg_id]

    def getRightBackLegIds(self):
        """Return the right back leg ids"""
        return self.legs[self.right_back_leg_id]

    def getRightBackFootId(self):
        """Return the right back foot id"""
        return self.feet[self.right_back_leg_id]
