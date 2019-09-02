#!/usr/bin/env python
r"""Provide the contact (force normal) constraint.

The lower unilateral contact force constraint is given by :math:`0 \leq f^i_n` where :math:`f^i_n` is the normal
force with respect to the contact surface applied on the link in contact :math:`i` defined in the local frame.

The optimization variables are these contact forces :math:`f^i_n` expressed in the world frame, thus they are
rotated to their local frame. This formulation can be rewritten as a unilateral inequality constraint
:math:`b_l \leq A_{ineq} x` in QP, with :math:`x = f^w \in \mathbb{R}^{6N_c}` which is the concatenation of all
the contact force variables (one for each contact point) expressed in the world frame,
:math:`A_{ineq} = R^l_w \in \mathbb{R}^{6N_c \times 6N_c}` is the block diagonal matrix that rotates the force
variables expressed in the world frame :math:`w` to their respective local frame :math:`l`, and
:math:`b_l = [-\infty, -\infty, 0, -\infty, -\infty, -\infty] * N_c`, where :math:`N_c` is the total number of
contact points.

The implementation of this class is inspired by [1].

References:
    - [1] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
"""

import numpy as np
from scipy.linalg import block_diag

from pyrobolearn.priorities.constraints.constraint import LowerUnilateralConstraint, ForceConstraint
from pyrobolearn.utils.transformation import get_matrix_from_quaternion

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Songyan Xin (insight)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ContactConstraint(LowerUnilateralConstraint, ForceConstraint):
    r"""Contact force constraint

    The lower unilateral contact force constraint is given by :math:`0 \leq f^i_n` where :math:`f^i_n` is the normal
    force with respect to the contact surface applied on the link in contact :math:`i` defined in the local frame.

    The optimization variables are these contact forces :math:`f^i_n` expressed in the world frame, thus they are
    rotated to their local frame. This formulation can be rewritten as a unilateral inequality constraint
    :math:`b_l \leq A_{ineq} x` in QP, with :math:`x = f^w \in \mathbb{R}^{6N_c}` which is the concatenation of all
    the contact force variables (one for each contact point) expressed in the world frame,
    :math:`A_{ineq} = R^l_w \in \mathbb{R}^{6N_c \times 6N_c}` is the block diagonal matrix that rotates the force
    variables expressed in the world frame :math:`w` to their respective local frame :math:`l`, and
    :math:`b_l = [-\infty, -\infty, 0, -\infty, -\infty, -\infty] * N_c`, where :math:`N_c` is the total number of
    contact points.

    The implementation of this class is inspired by [1].

    References:
        - [1] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
    """

    def __init__(self, model, contacts=[]):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            contacts (list[int], list[str]): list of contact links (ids or names).
        """
        super(ContactConstraint, self).__init__(model)

        # set variables
        self.contacts = contacts
        self._vector = -np.infty * np.ones(6)
        self._vector[2] = 0

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
        rotations = []
        for contact in self.contacts:
            link = self.model.get_link_id(contact)
            rot = get_matrix_from_quaternion(self.model.get_orientation(link)).T  # (3,3)
            rotations.append(block_diag((rot, rot)))  # (6,6)
        self._A_ineq = block_diag(rotations)  # (M*6,M*6)
        self._b_lower_bound = np.concatenate([self._vector for _ in self.contacts])  # (M*6,)
