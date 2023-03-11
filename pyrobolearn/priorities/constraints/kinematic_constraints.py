#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the various kinematic constraints used in QP.

References:
    [1] "Quadratic Programming in Python" (https://scaron.info/blog/quadratic-programming-in-python.html), Caron, 2017
    [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    [3] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
"""

import rbdl
import numpy as np

from pyrobolearn.robots.robot import Robot
from pyrobolearn.priorities.constraints.constraint import Constraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman and Alessio Rocchi)", "Songyan Xin"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class KinematicConstraint(Constraint):
    r"""Kinematic constraint

    """
    pass


class CartesianPose(KinematicConstraint):
    r"""Cartesian Pose constraint.

    """
    pass


class CartesianVelocity(KinematicConstraint):
    r"""Cartesian velocity constraint.

    """
    pass


class CoMVelocity(KinematicConstraint):
    r"""Center-of-mass velocity constraint.

    """
    pass


class JointLimits(KinematicConstraint):
    r"""Joint limits constraint.

    """
    pass


class JointVelocityLimits(KinematicConstraint):
    r"""Velocity limits constraint.

    """
    pass


class SelfCollisionAvoidance(KinematicConstraint):
    r"""Self-collision avoidance constraint.

    """
    pass


class ConvexHull(Constraint):
    r"""Convex Hull constraint

    """
    pass
