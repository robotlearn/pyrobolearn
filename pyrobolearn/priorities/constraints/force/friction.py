#!/usr/bin/env python
r"""Provide the friction cone (nonlinear) and pyramid (linear) constraint.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
"""

import numpy as np
from scipy.linalg import block_diag

from pyrobolearn.priorities.constraints.constraint import UpperUnilateralConstraint, ForceConstraint
from pyrobolearn.utils.transformation import get_matrix_from_quaternion

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Alessio Rocchi (C++)", "Songyan Xin (insight)",
               "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class FrictionConeConstraint(UpperUnilateralConstraint, ForceConstraint):
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

    def __init__(self, model, mu=0.7):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            mu (float): friction coefficient.
        """
        super(FrictionConeConstraint, self).__init__(model)

        # set variable
        self.mu = mu


class FrictionPyramidConstraint(UpperUnilateralConstraint, ForceConstraint):
    r"""Friction Pyramid constraint

    The friction pyramid constraint is a linear approximation of the friction cone.

    The friction pyramid is defined as:

    .. math:: P^i_s = {(f^i_x, f^i_y, f^i_z) \in \mathbb{R}^3 | |f^i_x| \leq \mu_i f^i_z, |f^i_y| \leq \mu_i f^i_z}

    where where :math:`i` denotes the ith support/contact, :math:`f^i_s` is the contact spatial force exerted at
    the contact point :math:`C_i`, and :math:`\mu_i` is the static friction coefficient at that contact point.
    If the static friction coefficient is given by :math:`\frac{\mu_i}{\sqrt{2}}`, then we are making an inner
    approximation (i.e. the pyramid is inside the cone) instead of an outer approximation (i.e. the cone is inside
    the pyramid). [1]

    This linear approximation is often used as a linear constraint in a quadratic optimization problem along with
    the unilateral constraint :math:`f^i_z \geq 0`.

    QP formulation
    --------------

    The friction pyramid constraints given by:

    .. math::

        -\mu f^i_z \leq f^i_x \leq \mu f^i_z \\
        -\mu f^i_z \leq f^i_y \leq \mu f^i_z

    can be rewritten as:

    .. math::

        f^i_x - \mu f^i_z \leq 0 \\
        -f^i_x - \mu f^i_z \leq 0 \\
        f^i_y - \mu f^i_z \leq 0 \\
        -f^i_y - \mu f^i_z \leq 0

    Thus, it can be rewritten as the inequality constraint :math:`G x \leq h` in QP, with:

    .. math::

        G = \left[\begin{array}{cccccc}
            1 & 0 & -\mu & 0 & 0 & 0  \\
            -1 & 0 & -\mu & 0 & 0 & 0  \\
            0 & 1 & -\mu & 0 & 0 & 0  \\
            0 & -1 & -\mu & 0 & 0 & 0  \\
        \end{array}\right],

    :math:`x = [f^i_x, f^i_y, f^i_z, n^i_x, n^i_y, n^i_z]` being the optimized variables, and :math:`h = [0,0,0,0]`.
    Note that the optimized variables are expressed in the world frame, and thus a rotation to express them in their
    contact local frame is performed beforehand.

    References:
        - [1] https://scaron.info/teaching/friction-cones.html
        - [2] "Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone
            for Rectangular Support Areas", Caron et al., 2015
        - [3] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
        - [4] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, mu=0.7, contacts=[]):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            mu (float): friction coefficient.
            contacts (list[int], list[str]): list of contact link unique id or name.
        """
        super(FrictionPyramidConstraint, self).__init__(model)

        # set variable
        self.mu = mu
        self._friction_matrix = np.zeros((4, 6))
        self._friction_matrix[0, 0] = 1
        self._friction_matrix[1, 0] = -1
        self._friction_matrix[2, 1] = 1
        self._friction_matrix[3, 1] = -1
        self._friction_matrix[:, 2] = -self.mu

        self.contacts = contacts

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def contacts(self):
        """Get the list of contact links."""
        return self._contacts

    @contacts.setter
    def contacts(self, contacts):
        """Set the contact links."""
        if not isinstance(contacts, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'contacts' to be a list of int/str, but instead got: "
                            "{}".format(type(contacts)))
        self._contacts = contacts

    ###########
    # Methods #
    ###########

    def _update(self):
        """Update the lower unilateral inequality matrix and vector."""
        self._A_ineq = np.zeros(4 * len(self.contacts), 6 * len(self.contacts))
        for i, contact in enumerate(self.contacts):
            rot = get_matrix_from_quaternion(self.model.get_orientation(self._link)).T
            rot = block_diag((rot, rot))
            self._A_ineq[i*4:(i+1)*4, i*6:(i+1)*6] = self._friction_matrix.dot()

        self._b_upper_bound = np.zeros(4 * len(self.contacts))
