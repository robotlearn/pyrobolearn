#!/usr/bin/env python
r"""Provide the joint position limits constraint.


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


class JointPositionLimitsConstraint(BoundConstraint, JointVelocityConstraint):
    r"""Joint Position Limits constraint.

    This provides bounds/limits on the joint positions, which are given by:

    .. math:: q_{lb} \leq q + \dot{q} dt \leq q_{ub}

    where :math:`(q_{lb}, q_{ub})` are the lower and upper bound of the joint positions respectively, :math:`q` are
    the current joint positions, :math:`\dot{q}` are the joint velocities being optimized, and :math:`dt` is the
    integration time step.

    This formulation can be rewritten as the inequality constraint :math:`Gx \leq h` used in QP, with
    :math:`G = [-dt*I, dt*I]^\top` and :math:`h = [(q - q_{lb})^\top, (q_{ub} - q)^\top]^\top` where :math:`I` is the
    square identity matrix.
    """

    def __init__(self, model, dt):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            dt (float): integration time step: use to compute :math:`q = q + \dot{q} dt`.
        """
        super(JointPositionLimitsConstraint, self).__init__(model)

        self.dt = dt

        bounds = self.model.get_joint_bounds()
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]

    def update(self):
        r"""
        Update the bounds.
        """
        q = self.model.get_joint_positions()
        self.lower_bound = 0
        self.upper_bound = 0
