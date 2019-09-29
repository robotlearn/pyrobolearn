# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the discrete dynamic movement primitive.
"""

import numpy as np

from pyrobolearn.models.dmp.canonical_systems import DiscreteCS
from pyrobolearn.models.dmp.forcing_terms import DiscreteForcingTerm
from pyrobolearn.models.dmp.dmp import DMP

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DiscreteDMP(DMP):
    r"""Discrete Dynamic Movement Primitive

    Discrete DMPs have the same mathematical formulation as general DMPs, which is given by:

    .. math:: \tau^2 \ddot{y} = K (g - y) - D \tau \dot{y} + f(s) (g - y0)

    where :math:`\tau` is a scaling factor that allows to slow down or speed up the reproduced movement, :math:`K`
    is the stiffness coefficient, :math:`D` is the damping coefficient, :math:`y, \dot{y}, \ddot{y}` are the position,
    velocity, and acceleration of a DoF, and :math:`f(s)` is the non-linear forcing term.

    However, the forcing term in the case of discrete DMPs is given by:

    .. math:: f(s) = \frac{\sum_i \psi_i(s) w_i}{\sum_i \psi_i(s)} s

    where :math:`w` are the learnable weight parameters, and :math:`\psi` are the basis functions evaluated at the
    given input phase variable :math:`s`, :math:`g` is the goal, and :math:`y_0` is the initial position. Note that
    as the phase converges to 0, the forcing term also converges to that value.

    The basis functions (in the discrete case) are given by:

    .. math:: \psi_i(s) = \exp \left( - \frac{1}{2 \sigma_i^2} (x - c_i)^2 \right)

    where :math:`c_i` is the center of the basis function :math:`i`, and :math:`\sigma_i` is its width.

    Also, the canonical system associated with this transformation system is given by:

    .. math:: \tau \dot{s} = - \alpha_s s

    where :math:`\tau` is a scaling factor that allows to slow down or speed up the movement, :math:`s` is the phase
    variable that drives the DMP, and :math:`\alpha_s` is a predefined constant.

    All these differential equations are solved using Euler's method.

    References:
        [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
    """

    def __init__(self, num_dmps, num_basis, dt=0.01, y0=0, goal=1,
                 forcing_terms=None, stiffness=None, damping=None):
        """Initialize the discrete DMP

        Args:
            num_dmps (int): number of DMPs
            num_basis (int, int[M]): number of basis functions, or list of number of basis functions.
            dt (float): step integration for Euler's method
            y0 (float, float[M]): initial position(s)
            goal (float, float[M]): goal(s)
            forcing_terms (list, ForcingTerm): the forcing terms (which can have different basis functions)
            stiffness (float): stiffness coefficient
            damping (float): damping coefficient
        """

        # create discrete canonical system
        cs = DiscreteCS(dt=dt)

        # create forcing terms (each one contains the basis functions and learnable weights)
        if forcing_terms is None:
            if isinstance(num_basis, int):
                forcing_terms = [DiscreteForcingTerm(cs, num_basis) for _ in range(num_dmps)]
            else:
                if not isinstance(num_basis, (np.ndarray, list, tuple, set)):
                    raise TypeError("Expecting 'num_basis' to be an int, list, tuple, np.array or set.")
                if len(num_basis) != num_dmps:
                    raise ValueError("The length of th list of number of basis doesn't match the number of DMPs")
                forcing_terms = [DiscreteForcingTerm(cs, n_basis) for n_basis in num_basis]

        # call super class constructor
        super(DiscreteDMP, self).__init__(canonical_system=cs, forcing_term=forcing_terms, y0=y0, goal=goal,
                                          stiffness=stiffness, damping=damping)

    def get_scaling_term(self, new_goal=None):
        """
        Return the scaling term for the forcing term.

        Args:
            new_goal (float, float[M], None): the new goal position. If None, it will be the current goal.

        Returns:
            float, float[M]: scaling term
        """
        if new_goal is None:
            new_goal = self.goal
        return (new_goal - self.y0) / (self.goal - self.y0)

    def _generate_goal(self, y_des):
        """Generate the goal for path imitation.

        Args:
            y_des (np.array): the desired trajectory to follow with shape [num_dmps, timesteps]

        Returns:
            float[M]: goal position
        """
        return np.copy(y_des[:, -1])


# Tests
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # tests canonical systems
    discrete_cs = DiscreteCS()

    # tests basis functions
    num_basis = 20
    discrete_f = DiscreteForcingTerm(discrete_cs, num_basis)

    # tests forcing terms
    f = np.sin(np.linspace(0, 2*np.pi, 100))
    discrete_f.train(f, plot=True)

    # Test discrete DMP
    discrete_dmp = DiscreteDMP(num_dmps=1, num_basis=num_basis)
    t = np.linspace(-6, 6, 100)
    y_target = 1 / (1 + np.exp(-t))
    discrete_dmp.imitate(y_target)
    y, dy, ddy = discrete_dmp.rollout()

    plt.plot(y_target, label='y_target')
    plt.plot(y[0], label='y_pred')
    # plt.plot(dy[0])
    # plt.plot(ddy[0])
    y, dy, ddy = discrete_dmp.rollout(new_goal=np.array([2.]))
    plt.plot(y[0], label='y_scaled')
    plt.title('Discrete DMP')
    plt.legend()
    plt.show()
