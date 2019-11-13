#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the Center of Mass velocity constraint.

The bilateral inequality CoM velocity constraint is given by:

.. math:: v_{lb} \leq J_{CoM}(q) \dot{q} \leq v_{ub}

where :math:`v_{lb}, v_{ub}` are the lower and upper bound of the CoM velocities, :math:`\dot{q}` are the joint
velocities being optimized, and :math:`J_{CoM}(q)` is the CoM Jacobian.

This formulation can be rewritten as a bilateral inequality constraint :math:`b_l \leq A_{ineq} x \leq b_u` in QP,
with :math:`x = \dot{q}`, :math:`A_{ineq} = J_{CoM}(q)`, :math:`b_l = v_{lb}` and :math:`b_u = v_{ub}`.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import BilateralConstraint, JointVelocityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CoMVelocityConstraint(BilateralConstraint, JointVelocityConstraint):
    r"""Center of Mass (CoM) Velocity constraint.

    The bilateral inequality CoM velocity constraint is given by:

    .. math:: v_{lb} \leq J_{CoM}(q) \dot{q} \leq v_{ub}

    where :math:`v_{lb}, v_{ub}` are the lower and upper bound of the CoM linear velocities, :math:`\dot{q}` are the
    joint velocities being optimized, and :math:`J_{CoM}(q)` is the CoM Jacobian.

    This formulation can be rewritten as a bilateral inequality constraint :math:`b_l \leq A_{ineq} x \leq b_u` in QP,
    with :math:`x = \dot{q}`, :math:`A_{ineq} = J_{CoM}(q)`, :math:`b_l = v_{lb}` and :math:`b_u = v_{ub}`.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, velocity_bounds):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            velocity_bounds (tuple[2 * np.array[float[3]]], np.array[float[3]], float, None): If tuple, it is the
              lower and upper bounds on the linear velocity. If np.array or float, then the lower and upper bound
              would be set to (-velocity_bounds, velocity_bounds). If None, it will not be considered.
        """
        super(CoMVelocityConstraint, self).__init__(model)

        # define variables
        self.velocity_bounds = velocity_bounds

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def velocity_bounds(self):
        """Get the CoM linear velocity bounds."""
        return self._vel_bounds

    @velocity_bounds.setter
    def velocity_bounds(self, bounds):
        """Set the CoM linear velocity bounds."""
        if bounds is not None:
            if isinstance(bounds, tuple):
                if len(bounds) != 2:
                    raise ValueError("Expecting the bounds to be a tuple of length 2, but got a length of "
                                     "{}".format(len(bounds)))
                for i, bound in enumerate(bounds):
                    if isinstance(bound, (int, float)):
                        bound = np.ones(3) * bound
                        bounds[i] = bound
                    elif isinstance(bound, np.ndarray) and len(bound) != 3:
                        raise ValueError("Expecting the given bound to be of length 3, but instead got a length of "
                                         "{}".format(len(bound)))
                    else:
                        raise TypeError("Expecting the given bound to be a np.array, but got instead: "
                                        "{}".format(type(bound)))
            elif isinstance(bounds, np.ndarray):
                bounds = bounds.reshape(-1)
                if len(bounds) == 3:
                    bounds = (-bounds, bounds)
                elif len(bounds) == 6:
                    bounds = (bounds[:3], bounds[3:])
                else:
                    raise ValueError("Expecting the given bounds to be of length 3 or 6 but got instead a length of: "
                                     "{}".format(len(bounds)))
            elif isinstance(bounds, (int, float)):
                bounds = (-np.ones(3) * bounds, np.ones(3) * bounds)
            else:
                raise TypeError("Expecting the given bounds to be a tuple of np.array, a np.array, or None, but "
                                "instead got: {}".format(type(bounds)))
        self._vel_bounds = bounds

    ###########
    # Methods #
    ###########

    def _update(self):
        """Update the inequality matrix and vectors."""
        self._A_ineq = self.model.get_com_jacobian(full=False)  # (3,N)
        self._b_lower_bound, self._b_upper_bound = self._vel_bounds
