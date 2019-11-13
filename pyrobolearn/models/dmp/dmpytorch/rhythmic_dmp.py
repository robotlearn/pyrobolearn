#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the rhythmic dynamic movement primitive.
"""

import numpy as np
import torch

from pyrobolearn.models.dmp.dmpytorch.canonical_systems import RhythmicCS
from pyrobolearn.models.dmp.dmpytorch.forcing_terms import RhythmicForcingTerm
from pyrobolearn.models.dmp.dmpytorch.dmp import DMP

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RhythmicDMP(DMP):
    r"""Rhythmic Dynamic Movement Primitive

    Rhythmic DMPs have the same mathematical formulation as general DMPs, which is given by:

    .. math:: \tau^2 \ddot{y} = K (g - y) - D \tau \dot{y} + f(s)

    where :math:`\tau` is a scaling factor that allows to slow down or speed up the reproduced movement, :math:`K`
    is the stiffness coefficient, :math:`D` is the damping coefficient, :math:`y, \dot{y}, \ddot{y}` are the position,
    velocity, and acceleration of a DoF, and :math:`f(s)` is the non-linear forcing term.

    However, the forcing term in the case of rhythmic DMPs is given by:

    .. math:: f(s) = \frac{\sum_i \psi_i(s) w_i}{\sum_i \psi_i(s)} a

    where :math:`w` are the learnable weight parameters, and :math:`\psi` are the basis functions evaluated at the
    given input phase variable :math:`s`, and :math:`a` is the amplitude.

    The basis functions (in the rhythmic case) are given by:

    .. math:: \psi_i(s) = \exp \left( - h_i (\cos(s - c_i) - 1) \right)

    where :math:`c_i` is the center of the basis, and :math:`h_i` is a measure of concentration.

    Also, the canonical system associated with this transformation system is given by:

    .. math:: \tau \dot{s} = 1

    where :math:`\tau` is a scaling factor that allows to slow down or speed up the movement, and :math:`s` is the
    phase variable that drives the DMP.

    All these differential equations are solved using Euler's method.

    References:
        [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
    """

    def __init__(self, num_dmps, num_basis, dt=0.01, y0=0, goal=1,
                 forces=None, stiffness=None, damping=None):
        """Initialize the rhythmic DMP

        Args:
            num_dmps (int): number of DMPs
            num_basis (int): number of basis functions
            dt (float): step integration for Euler's method
            y0 (float, np.array): initial position(s)
            goal (float, np.array): goal(s)
            forces (list, ForcingTerm): the forcing terms (which can have different basis functions)
            stiffness (float): stiffness coefficient
            damping (float): damping coefficient
        """

        # create rhythmic canonical system
        cs = RhythmicCS(dt=dt)

        # create forcing terms (each one contains the basis functions and learnable weights)
        if forces is None:
            if isinstance(num_basis, int):
                forces = [RhythmicForcingTerm(cs, num_basis) for _ in range(num_dmps)]
            else:
                if not isinstance(num_basis, (np.ndarray, list, tuple, set)):
                    raise TypeError("Expecting 'num_basis' to be an int, list, tuple, np.array or set.")
                if len(num_basis) != num_dmps:
                    raise ValueError("The length of th list of number of basis doesn't match the number of DMPs")
                forces = [RhythmicForcingTerm(cs, n_basis) for n_basis in num_basis]

        # call super class constructor
        super(RhythmicDMP, self).__init__(canonical_system=cs, forces=forces, y0=y0, goal=goal,
                                          stiffness=stiffness, damping=damping)

    def get_scaling_term(self, new_goal=None):
        """
        Return the scaling term for the forcing term. For rhythmic DMPs it's non-diminishing, so this function just
        returns 1.
        """
        return torch.ones(self.num_dmps)

    def _generate_goal(self, y_des):
        """Generate the goal for path imitation.

        For rhythmic DMPs, the goal is the average of the desired trajectory.

        Args:
            y_des (float[M,T]): the desired trajectory to follow (with shape [num_dmps, timesteps])

        Returns:
            float[M]: goal positions (one for each DMP)
        """
        goal = np.zeros(self.num_dmps)
        for n in range(self.num_dmps):
            num_idx = ~torch.isnan(y_des[n])  # ignore nan's when calculating goal
            goal[n] = .5 * (y_des[n, num_idx].min() + y_des[n, num_idx].max())
        return goal
