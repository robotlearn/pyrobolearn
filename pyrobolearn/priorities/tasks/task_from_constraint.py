# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Transform a given equality constraint into a task.

An equality constraint specified by :math:`Fx = k` is transformed to a soft task :math:`||Ax - b||^2`, where
:math:`A = F` and :math:`b = k`. This allows for the equality constraint to be lightly violated; by specifying the
weight :math:`W` we can specify how much the constraint should be satisfied.
"""

from pyrobolearn.priorities.tasks import Task
from pyrobolearn.priorities.constraints import EqualityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TaskFromConstraint(Task):
    r"""Task From Equality Constraint

    An equality constraint specified by :math:`Fx = k` is transformed to a soft task :math:`||Ax - b||_{W}^2`, where
    :math:`A = F` and :math:`b = k`. This allows for the equality constraint to be lightly violated; by specifying the
    weight :math:`W` we can specify how much the constraint should be satisfied.
    """

    def __init__(self, constraint, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            constraint (EqualityConstraint): equality constraint.
            weight (float, np.array[float[6,6]], np.array[float[3,3]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        # set equality constraint
        self.equality_constraint = constraint

        # call superclass
        super(TaskFromConstraint, self).__init__(model=constraint.model, weight=weight, constraints=constraints)

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def equality_constraint(self):
        """Get the equality constraint."""
        return self._constraint

    @equality_constraint.setter
    def equality_constraint(self, constraint):
        """Set the equality constraint."""
        if not isinstance(constraint, EqualityConstraint):
            raise TypeError("Expecting the given 'constraint' to be an instance of `EqualityConstraint`, but "
                            "instead got: {}".format(constraint))
        self._constraint = constraint

    ###########
    # Methods #
    ###########

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        # update equality constraint
        self.equality_constraint.update()

        # update A and b
        self._A = self.equality_constraint.A_eq
        self._b = self.equality_constraint.b_eq
