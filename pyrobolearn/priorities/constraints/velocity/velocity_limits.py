#!/usr/bin/env python
r"""Provide the velocity limits constraint.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import BoundConstraint, JointVelocityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointVelocityLimitsConstraint(BoundConstraint, JointVelocityConstraint):
    r"""Joint velocity limits constraint.

    This provides bounds/limits on the joint velocities

    .. math::  \dot{q}_{lb} \leq \dot{q} \leq \dot{q}_{ub}

    where :math:`(\dot{q}_{lb}, \dot{q}_{ub})` are the lower and upper bound on the joint velocities, and
    :math:`\dot{q}` are the joint velocities being optimized.

    This formulation can be rewritten as the inequality constraint :math:`Gx \leq h` used in QP, with
    :math:`G = [-I, I]^\top` and :math:`h = [-q_{lb}^\top, q_{ub}^\top]^\top` where :math:`I` is the square identity
    matrix.
    """

    def __init__(self, model):
        """
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
        """
        super(JointVelocityLimitsConstraint, self).__init__(model)

        bounds = self.model.get_joint_velocity_bounds()
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]

    def update(self):
        pass
