#!/usr/bin/env python
r"""Provide the dynamic feasibility constraint.

The equality dynamic feasibility constraint is given by:

.. math:: H(q) \ddot{q} + N(q, \dot{q}) = \sum_i J_i^T F_i

where :math:`H(q)` is the joint space inertia matrix, :math:`\ddot{q}` are the joint accelerations being optimized,
:math:`N(q, \dot{q})` is the vector of force terms that account for the Coriolis and centrifugal forces, gravity,
and any other forces acting on the system other than the contact forces given by :math:`\sum_i J_i^T F_i` (where
each :math:`J_i` is a Jacobian matrix and :math:`F_i` is a wrench vector at the contact link :math:`i`).

This formulation can be rewritten as an equality constraint math:`A_{eq} x = b_{eq}` in QP, with
:math:`x = \ddot{q}`, :math:`A_{eq} = H(q)`, and :math:`b_{eq} = \sum_i J_i^T F_i - N(q, \dot{q})`.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import EqualityConstraint, JointAccelerationConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Arturo Laurenzi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DynamicFeasibilityConstraint(EqualityConstraint, JointAccelerationConstraint):
    r"""Dynamic Feasibility Constraint

    The equality joint acceleration constraint is given by:

    .. math:: H(q) \ddot{q} + N(q, \dot{q}) = \sum_i J_i^T F_i

    where :math:`H(q)` is the joint space inertia matrix, :math:`\ddot{q}` are the joint accelerations being optimized,
    :math:`N(q, \dot{q})` is the vector of force terms that account for the Coriolis and centrifugal forces, gravity,
    and any other forces acting on the system other than the contact forces given by :math:`\sum_i J_i^T F_i` (where
    each :math:`J_i` is a Jacobian matrix and :math:`F_i` is a wrench vector at the contact link :math:`i`).

    This formulation can be rewritten as an equality constraint math:`A_{eq} x = b_{eq}` in QP, with
    :math:`x = \ddot{q}`, :math:`A_{eq} = H(q)`, and :math:`b_{eq} = \sum_i J_i^T F_i - N(q, \dot{q})`.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, contact_links=[], wrenches=[]):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            contact_links (list[str], list[int], None): list of unique contact link names or ids.
            wrenches (list[np.array[float[6]]], None): list of associated wrenches applied to the contact links. It
              must have the same size as the number of contact links.
        """
        super(DynamicFeasibilityConstraint, self).__init__(model)

        # set variables
        self.contact_links = contact_links
        self.wrenches = wrenches

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def contact_links(self):
        """Get the contact links."""
        return self._contact_links

    @contact_links.setter
    def contact_links(self, contacts):
        """Set the contact links."""
        if contacts is None:
            contacts = []
        elif not isinstance(contacts, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'contact_links' to be a list of names/ids, but got instead: "
                            "{}".format(type(contacts)))
        self._contact_links = contacts

    @property
    def wrenches(self):
        """Get the wrenches."""
        return self._wrenches

    @wrenches.setter
    def wrenches(self, wrenches):
        """Set the wrenches."""
        if wrenches is None:
            wrenches = []
        elif not isinstance(wrenches, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'wrenches' to be a list of wrench vectors, but got instead: "
                            "{}".format(type(wrenches)))
        if isinstance(wrenches, np.ndarray) and wrenches.ndim == 1:
            wrenches = wrenches.reshape(-1, 6)
        self._wrenches = wrenches

    ###########
    # Methods #
    ###########

    def _update(self):
        """Update the equality constraint."""
        # get dynamics
        H = self.model.get_inertia_matrix()
        N = self.model.compute_nonlinear_term()

        # compute torques due to contact wrenches
        tau = 0
        for link, wrench in zip(self.contact_links, self.wrenches):
            link = self.model.get_link_id(link)
            jacobian = self.model.get_jacobian(link=link)
            tau += jacobian.T.dot(wrench)

        # equality constraints
        self._A_eq = H
        self._b_eq = tau - N
