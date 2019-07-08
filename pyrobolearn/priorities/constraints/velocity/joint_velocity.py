#!/usr/bin/env python
r"""Provide the differential kinematics constraint.

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import EqualityConstraint, JointVelocityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DifferentialKinematicsConstraint(EqualityConstraint, JointVelocityConstraint):
    r"""Differential kinematics constraint.

    This provides the joint velocity constraints which is given by:

    .. math:: J(q) \dot{q} = v

    where :math:`J(q)` is the jacobian from a base link to a distal link, :math:`\dot{q}` are the joint velocities
    being optimized, and :math:`v` is the imposed cartesian velocity imposed on the distal link.

    This formulation can be rewritten as the inequality constraint :math:`A_{eq} x = b_{eq}` used in QP, with
    :math:`A_{eq} = J(q)`, :math:`x = \dot{q}`, and :math:`b_{eq} = v`.
    """

    def __init__(self, model, distal_link, base_link=None, velocity=None):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            distal_link (int, str): distal link id or name.
            base_link (int, str, None): base link id or name. If None, it will be the world.
            velocity (np.array[6], None): imposed velocity. If None, it will be set to 0.
        """
        super(DifferentialKinematicsConstraint, self).__init__(model)
        raise NotImplementedError

    def update(self):
        r"""
        Update the bounds.
        """
        raise NotImplementedError
