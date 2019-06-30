#!/usr/bin/env python
r"""Provide the friction cone (nonlinear) and pyramid (linear) constraint.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import Constraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Alessio Rocchi (C++)", "Songyan Xin (insight)",
               "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class FrictionConeConstraint(Constraint):
    r"""Friction Cone constraint

    The friction cone is defined as:

    .. math:: C^i_s = {(f^i_x, f^i_y, f^i_z) \in \mathbb{R}^3 | \sqrt{(f^i_x)^2 + (f^i_y)^2} \leq \mu_i f^i_z }

    where :math:`i` denotes the ith support/contact, :math:`f^i_s` is the contact spatial force exerted at
    the contact point :math:`C_i`, and :math:`\mu_i` is the static friction coefficient at that contact point.

    "A point contact remains in the fixed contact mode while its contact force f^i lies inside the friction cone"
    [1]. Often, the friction pyramid which is the linear approximation of the friction cone is considered as it
    is easier to manipulate it; e.g. present it as a linear constraint in a quadratic optimization problem.

    References:
        - [1] https://scaron.info/teaching/friction-cones.html
        - [2] "Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone
            for Rectangular Support Areas", Caron et al., 2015
        - [3] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
        - [4] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
    """

    def __init__(self, model):
        super(FrictionConeConstraint, self).__init__(model)


class FrictionPyramidConstraint(Constraint):
    r"""Friction Pyramid constraint

    The friction pyramid constraint is a linear approximation of the friction cone.

    The friction pyramid is defined as:

    .. math:: P^i_s = {(f^i_x, f^i_y, f^i_z) \in \mathbb{R}^3 | f^i_x \leq \mu_i f^i_z, f^i_y \leq \mu_i f^i_z}

    where where :math:`i` denotes the ith support/contact, :math:`f^i_s` is the contact spatial force exerted at
    the contact point :math:`C_i`, and :math:`\mu_i` is the static friction coefficient at that contact point.
    If the static friction coefficient is given by :math:`\frac{\mu_i}{\sqrt{2}}`, then we are making an inner
    approximation (i.e. the pyramid is inside the cone) instead of an outer approximation (i.e. the cone is inside
    the pyramid). [1]

    This linear approximation is often used as a linear constraint in a quadratic optimization problem along with
    the unilateral constraint :math:`f^i_z \geq 0`.

    References:
        - [1] https://scaron.info/teaching/friction-cones.html
        - [2] "Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone
            for Rectangular Support Areas", Caron et al., 2015
    """

    def __init__(self, model):
        super(FrictionPyramidConstraint, self).__init__(model)