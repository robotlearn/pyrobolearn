# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the wrench limits constraint.

This provides bounds/limits on the wrenches:

.. math::  F_{lb} \leq F \leq F_{ub}

where :math:`(F_{lb}, F_{ub})` are the lower and upper bound on the wrenches, and
:math:`F` is the wrench vector being optimized.

This formulation can be rewritten as the inequality constraint math:`lb \leq x \leq ub` in QP, with
:math:`lb = F_{lb}`, :math:`ub = F_{ub}`, and :math:`x = F`. This can also be rewritten as :math:`Gx \leq h`,
with :math:`G = [-I, I]^\top` and :math:`h = [-F_{lb}^\top, F_{ub}^\top]^\top` where :math:`I` is the square
identity matrix.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import BoundConstraint, ForceConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WrenchLimitsConstraint(BoundConstraint, ForceConstraint):
    r"""Wrench Limits constraint.

     This provides bounds/limits on the wrenches:

    .. math::  F_{lb} \leq F \leq F_{ub}

    where :math:`(F_{lb}, F_{ub})` are the lower and upper bound on the wrenches, and
    :math:`F` is the wrench vector being optimized.

    This formulation can be rewritten as the inequality constraint math:`lb \leq x \leq ub` in QP, with
    :math:`lb = F_{lb}`, :math:`ub = F_{ub}`, and :math:`x = F`. This can also be rewritten as :math:`Gx \leq h`,
    with :math:`G = [-I, I]^\top` and :math:`h = [-F_{lb}^\top, F_{ub}^\top]^\top` where :math:`I` is the square
    identity matrix.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, bounds):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            bounds (tuple[2 * np.array[float[M]]], np.array[float[M]]): wrench limits, where `M` is 3 (vector of
              forces) or 6 (vector of forces and torques). If tuple, it is the lower and upper bounds on the wrenches.
              If np.array, then the lower and upper bound will be set to (-bounds, bounds).
        """
        super(WrenchLimitsConstraint, self).__init__(model)

        # set variables
        self.bounds = bounds

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def bounds(self):
        """Get the wrench bounds."""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """Set the wrench bounds."""
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
                elif len(bounds) == 12:
                    bounds = (bounds[:6], bounds[6:])
                else:
                    raise ValueError("Expecting the given bounds to be of length 3 or 6 but got instead a length of: "
                                     "{}".format(len(bounds)))
            elif isinstance(bounds, (int, float)):
                bounds = (-np.ones(3) * bounds, np.ones(3) * bounds)
            else:
                raise TypeError("Expecting the given bounds to be a tuple of np.array, a np.array, or None, but "
                                "instead got: {}".format(type(bounds)))
        self._bounds = bounds

    ###########
    # Methods #
    ###########

    def _update(self):
        """Update the lower and upper bounds."""
        self._b_lower_bound, self._b_upper_bound = self._bounds
