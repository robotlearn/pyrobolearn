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
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DynamicConstraint(Constraint):
    r"""Dynamic Constraints

    Dynamic constraint using Equation of Motion (by using inertia and the non-linear terms).
    """
    pass


class StaticsStability(DynamicConstraint):
    r"""Statics Stability constraint.

    'The goal of statics is to determine the relationship between the generalized forces applied to the end-effector
    and the generalized forces applied to the joints.' [1]

    References:
        [1] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
    """
    pass


class TorqueLimits(DynamicConstraint):
    r"""Torque limits constraint.

    """
    pass


class FrictionCones(DynamicConstraint):
    r"""Friction cones constraint (using contact force optimization)

    """
    pass


class FrictionPyramide(DynamicConstraint):
    r"""Friction pyramide constraint (using contact force optimization)
    
    """
    pass


class UnilateralContact(DynamicConstraint):
    r"""Unilateral Contact constraint.

    Mechanical constraint which prevents penetration between two bodies.
    """
    pass


class WrenchLimits(DynamicConstraint):
    r"""Wrench Limits (using contact force optimization).

    """
    pass


class ZMP(DynamicConstraint):
    r"""ZMP constraint.

    """
    pass
