#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the dynamic feasibility task (which is based on the dynamic feasibility constraint).

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

from pyrobolearn.priorities.tasks import JointAccelerationTask, TaskFromConstraint
from pyrobolearn.priorities.constraints.acceleration.dynamic_feasibility import DynamicFeasibilityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DynamicFeasibilityTask(JointAccelerationTask):
    r"""Dynamic Feasibility Task

    The dynamic feasibility constraint tries to enforce the joint space dynamic equation of motion given by
    :math:`H(q) \ddot{q} + N(q, \dot{q}) = \sum_i J_i^T F_i`. This is a softer version of the corresponding equality
    constraint (see `priorities/constraints/acceleration/dynamic_feasibility.py`).

    The task minimizes:

    .. math:: || H(q) \ddot{q} - (\sum_i J_i^T F_i - N(q, \dot{q})) ||^2,

    where :math:`H(q)` is the joint space inertia matrix, :math:`\ddot{q}` are the joint accelerations being optimized,
    :math:`N(q, \dot{q})` is the vector of force terms that account for the Coriolis and centrifugal forces, gravity,
    and any other forces acting on the system other than the contact forces given by :math:`\sum_i J_i^T F_i` (where
    each :math:`J_i` is a Jacobian matrix and :math:`F_i` is a wrench vector at the contact link :math:`i`).

    This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=H(q)`,
    :math:`x=\ddot{q}`, and :math:`b = \sum_i J_i^T F_i - N(q, \dot{q})`.

    Compared to the constraint, this task can be violated during the optimization. The user can set the weight to
    specify how much this task can be violated.
    """

    def __init__(self, model, contact_links=[], wrenches=[], weight=1., constraints=[]):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            contact_links (list[str], list[int], None): list of unique contact link names or ids.
            wrenches (list[np.array[float[6]]], None): list of associated wrenches applied to the contact links. It
              must have the same size as the number of contact links.
            weight (float, np.array[float[6,6]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(DynamicFeasibilityTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # set inner task based on constraint
        self._constraint = DynamicFeasibilityConstraint(model=model, contact_links=contact_links, wrenches=wrenches)
        self._task = TaskFromConstraint(self._constraint)

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def contact_links(self):
        """Get the contact links."""
        return self._constraint.contact_links

    @contact_links.setter
    def contact_links(self, contacts):
        """Set the contact links."""
        self._constraint.contact_links = contacts

    @property
    def wrenches(self):
        """Get the wrenches."""
        return self._constraint.wrenches

    @wrenches.setter
    def wrenches(self, wrenches):
        """Set the wrenches."""
        self._constraint.wrenches = wrenches

    ###########
    # Methods #
    ###########

    def _update(self, x=None):
        """Update the equality constraint."""
        self._A = self._task.A
        self._b = self._task.b
