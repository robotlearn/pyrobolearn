# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the cartesian velocity constraint.

The bilateral inequality cartesian velocity constraint is given by:

.. math:: v_{lb} \leq J(q) \dot{q} \leq v_{ub}

where :math:`v_{lb}, v_{ub}` are the lower and upper bound on the cartesian velocities of a given distal link,
:math:`\dot{q}` are the joint velocities being optimized, and :math:`J(q)` is the Jacobian from the base to the
distal link.

This formulation can be rewritten as a bilateral inequality constraint :math:`b_l \leq A_{ineq} x \leq b_u` in QP,
with :math:`x = \dot{q}`, :math:`A_{ineq} = J(q)`, :math:`b_l = v_{lb}` and :math:`b_u = v_{ub}`.


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


class CartesianVelocityConstraint(BilateralConstraint, JointVelocityConstraint):
    r"""Cartesian Velocity constraint.

    The bilateral inequality cartesian velocity constraint is given by:

    .. math:: v_{lb} \leq J(q) \dot{q} \leq v_{ub}

    where :math:`v_{lb}, v_{ub}` are the lower and upper bound on the cartesian velocities of a given distal link,
    :math:`\dot{q}` are the joint velocities being optimized, and :math:`J(q)` is the Jacobian from the base to the
    distal link.

    This formulation can be rewritten as a bilateral inequality constraint :math:`b_l \leq A_{ineq} x \leq b_u` in QP,
    with :math:`x = \dot{q}`, :math:`A_{ineq} = J(q)`, :math:`b_l = v_{lb}` and :math:`b_u = v_{ub}`.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, distal_link, base_link=None, local_position=(0, 0, 0), linear_velocity_bounds=None,
                 angular_velocity_bounds=None):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            distal_link (int, str): distal link id or name.
            base_link (int, str, None): base link id or name. If None, it will be the world.
            local_position (np.array[float[3]]): local position on the distal link.
            linear_velocity_bounds (tuple[2 * np.array[float[3]]], np.array[float[3]], None): If tuple, it is the
              lower and upper bounds on the linear velocity. If np.array, then the lower and upper bound would be set
              to (-linear_velocity_bounds, linear_velocity_bounds). If None, it will not be considered.
            angular_velocity_bounds (tuple[2 * np.array[float[3]]], np.array[float[3]], None): If tuple, it is the
              lower and upper bounds on the angular velocity. If np.array, then the lower and upper bound would be set
              to (-angular_velocity_bounds, angular_velocity_bounds). If None, it will not be considered.
        """
        super(CartesianVelocityConstraint, self).__init__(model)

        # define variables
        self.distal_link = self.model.get_link_id(distal_link)
        self.base_link = self.model.get_link_id(base_link) if base_link is not None else base_link
        self.local_position = local_position
        
        self.linear_velocity_bounds = linear_velocity_bounds
        self.angular_velocity_bounds = angular_velocity_bounds
        
        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def linear_velocity_bounds(self):
        """Get the cartesian linear velocity bounds of the distal link wrt the base."""
        return self._lin_vel_bounds

    @linear_velocity_bounds.setter
    def linear_velocity_bounds(self, bounds):
        """Set the cartesian linear velocity bounds of the distal link wrt the base."""
        if bounds is not None:
            if isinstance(bounds, tuple):
                if len(bounds) != 2:
                    raise ValueError("Expecting the bounds to be a tuple of length 2, but got a length of "
                                     "{}".format(len(bounds)))
                for bound in bounds:
                    if not isinstance(bound, np.ndarray):
                        raise TypeError("Expecting the given bound to be a np.array, but got instead: "
                                        "{}".format(type(bound)))
                    if len(bound) != 3:
                        raise ValueError("Expecting the given bound to be of length 3, but instead got a length of "
                                         "{}".format(len(bound)))
            elif isinstance(bounds, np.ndarray):
                if len(bounds) == 3:
                    bounds = (-bounds, bounds)
                elif len(bounds) == 6:
                    bounds = (bounds[:3], bounds[3:])
            else:
                raise TypeError("Expecting the given bounds to be a tuple of np.array, a np.array, or None, but "
                                "instead got: {}".format(type(bounds)))
        self._lin_vel_bounds = bounds

    @property
    def angular_velocity_bounds(self):
        """Get the cartesian angular velocity bounds of the distal link wrt the base."""
        return self._ang_vel_bounds

    @angular_velocity_bounds.setter
    def angular_velocity_bounds(self, bounds):
        """Set the cartesian angular velocity bounds of the distal link wrt the base."""
        if bounds is not None:
            if isinstance(bounds, tuple):
                if len(bounds) != 2:
                    raise ValueError("Expecting the bounds to be a tuple of length 2, but got a length of "
                                     "{}".format(len(bounds)))
                for bound in bounds:
                    if not isinstance(bound, np.ndarray):
                        raise TypeError("Expecting the given bound to be a np.array, but got instead: "
                                        "{}".format(type(bound)))
                    if len(bound) != 3:
                        raise ValueError("Expecting the given bound to be of length 3, but instead got a length of "
                                         "{}".format(len(bound)))
            elif isinstance(bounds, np.ndarray):
                if len(bounds) == 3:
                    bounds = (-bounds, bounds)
                elif len(bounds) == 6:
                    bounds = (bounds[:3], bounds[3:])
            else:
                raise TypeError("Expecting the given bounds to be a tuple of np.array, a np.array, or None, but "
                                "instead got: {}".format(type(bounds)))
        self._ang_vel_bounds = bounds

    ###########
    # Methods #
    ###########

    def _update(self):
        """Update the inequality matrix and vectors."""
        if self.linear_velocity_bounds is None and self.angular_velocity_bounds is None:
            raise ValueError("Expecting at least the linear or angular velocity bounds to be specified, but instead "
                             "got None for both of them.")

        self._A_ineq = self.model.get_jacobian(link=self.distal_link, wrt_link=self.base_link,
                                               point=self.local_position)

        if self.linear_velocity_bounds is None:
            self._b_lower_bound, self._b_upper_bound = self.angular_velocity_bounds
            self._A_ineq = self._A_ineq[3:]
        elif self.angular_velocity_bounds is None:
            self._b_lower_bound, self._b_upper_bound = self.linear_velocity_bounds
            self._A_ineq = self._A_ineq[:3]
        else:
            b_lin_low, b_lin_up = self.linear_velocity_bounds
            b_ang_low, b_ang_up = self.angular_velocity_bounds
            self._b_lower_bound = np.concatenate((b_lin_low, b_ang_low))
            self._b_upper_bound = np.concatenate((b_lin_up, b_ang_up))
