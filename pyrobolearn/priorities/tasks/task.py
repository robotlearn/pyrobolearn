#!/usr/bin/env python
r"""Provide the various tasks (i.e. objective functions) used in QP.

The tasks presented here represents the quadratic objective functions used in quadratic programming (QP).

A quadratic program (QP) is written in standard form [1] as:

.. math::

    x^* &= \arg \min_x \frac{1}{2} x^T Q x + p^T x \\ \text{subj. to}
        & Gx \leq h \\
        & Fx = c

where :math:`x` is the vector being optimized (in robotics, it can be joint positions, velocities, torques, ...),
"the matrix :math:`Q` and vector :math:`p` are used to define any quadratic objective function of these variables,
while the matrix-vector couples :math:`(G,h)` and :math:`(F,c)` respectively define inequality and equality
constraints" [1]. Inequality constraints can include the lower bounds and upper bounds of :math`x` by setting
:math:`G` to be the identity matrix or minus this one, and :math:`h` to be the upper or lower bounds.

For instance, the quadratic objective function :math:`||Ax - b||^2_{W}` (where :math:`W` is a weight matrix) is given
in the standard form as:

.. math:: ||Ax - b||^2_{W} = (Ax - b)^\top W (Ax - b) = x^\top A^\top W A x - 2 b^\top W A x + b^\top W b

where the last term :math:`b^\top W b` can be removed as it does not depend on the variables we are optimizing (i.e.
:math:`x`). We thus have :math:`Q = A^\top W A` a symmetric matrix and :math:`p = -2 b^\top W A`.

Many control problems in robotics can be formulated as a quadratic programming problem.

For instance, let's assume that we want to optimize the joint velocities :math:`\dot{q}` given the end-effector's
desired position and velocity in task space. We can define the quadratic problem as:

.. math:: || J(q) \dot{q} - \dot{x} ) ||^2

where using a PD reference, :math:`\dot{x} = \dot{x}_d + K (x_d - x)`, where :math:`x_d` and :math:`x` are the desired
and current end-effector's position respectively, and :math:`\dot{x}_d` is the desired velocity.


* Soft priority tasks: with soft-priority tasks, the quadratic programming problem being minimized for n such tasks
is given by:

.. math::

    x^* &= \arg \min_x ||A_1 x - b_1||^2_{W_1} + ||A_2 x - b_2 ||^2_{W_2} + ... + ||A_n x - b_n ||^2_{W_n} \\
    \text{subj. to} & Gx \leq h \\
                    & Fx = c

Often, the weight matrices :math:`W_i` are just scalars :math:`w_i`. This problem can notably be solved by stacking
the :math:`A_i` one of top of another, and stacking the :math:`b_i` and :math:`W_i` in the same manner, and solving
:math:`||A x - b||^2_{W}` This is known as the augmented task. When the matrices :math:`A` are Jacobians this is known
as the augmented Jacobian (which can sometimes be ill-conditioned).

* Hard priority tasks: with hard-priority tasks, the quadratic programming problem for n tasks is defined in a
sequential manner, where the first most important task will be first optimized, and then the subsequent tasks will be
optimized one after the other. Thus, the first task to be optimized is given by:

.. math:: x_1^* &= \arg \min_x ||A_1 x - b_1||^2 \\ \text{subj. to}
                & G_1 x \leq h_1 \\
                & F_1 x = c_1,

while the second next most important task that would be solved is given by:

.. math:: x_2^* &= \arg \min_x ||A_2 x - b_2||^2 \\ \text{subj. to}
                & G_2 x \leq h_2 \\
                & F_2 x = c_2 \\
                & A_1 x = A_1 x_1^* \\
                & G_1 x \leq h_1 \\
                & F_1 x = c_1,

until the :math:`n` most important task, given by:

.. math:: x_n^* \arg \min_x ||A_n x - b_n||^2 \\ \text{subj. to}
                & A_1 x = A_1 x_1^* \\
                & ... \\
                & A_{n-1} x = A_{n-1} x_{n-1}^* \\
                & G_1 x \leq h_1 \\
                & ... \\
                & G_n x \leq h_n \\
                & F_1 x = c_1 \\
                & ... \\
                & F_n x = c_n.

By setting the previous :math:`A_{i-1} x = A_{i-1} x_{i-1}^*` as equality constraints, the current solution
:math:`x_i^*` won't change the optimality of all higher priority tasks.


References:
    [1] "Quadratic Programming in Python" (https://scaron.info/blog/quadratic-programming-in-python.html), Caron, 2017
    [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    [3] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import Constraint

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman and Alessio Rocchi)", "Songyan Xin"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: take into account constraints
# TODO: take into account hard priority tasks
class Task(object):
    r"""Task (abstract) class.

    Python implementation of Tasks based on the slides of the OpenSoT framework [1].

    References:
        [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN"
            ([code](https://opensot.wixsite.com/opensot),
            [slides](https://docs.google.com/presentation/d/1kwJsAnVi_3ADtqFSTP8wq3JOGLcvDV_ypcEEjPHnCEA),
            [tutorial video](https://www.youtube.com/watch?v=yFon-ZDdSyg),
            [old code](https://github.com/songcheng/OpenSoT)), Rocchi et al., 2015
    """

    def __init__(self, tasks=[], model=None, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            tasks (list of list of Task): list of list of tasks, where the list is ordered by hard priorities, and the
                nested list contains tasks which
            model (Robot, str): robot model. If str, it needs to be the path to the URDF.
            constraints (list of Constraint): list of constraints.
        """
        self._tasks = tasks
        self._model = model
        self.weight = weight
        self._constraints = []

        self._A = None
        self._b = None

    ##############
    # Properties #
    ##############

    @property
    def tasks(self):
        """Return the tasks."""
        return self._tasks

    @property
    def level(self):
        """Return the level of the tree."""
        if self._tasks:
            return len(self._tasks)
        else:
            return 1

    @property
    def model(self):
        """Return the robot model."""
        return self._model

    @property
    def weight(self):
        """Return the relative weight (used for soft priorities)."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        if not isinstance(weight, (int, float)):
            raise TypeError("Expecting the relative weight to be an int or float, instead got: {}".format(type(weight)))
        if weight < 0:
            raise ValueError("Expecting the relative weight to be positive.")
        self._weight = weight

    @property
    def constraints(self):
        """Return the constraints."""
        return self._constraints

    @property
    def A(self):
        """Return A matrix used in QP."""
        return self._A

    @property
    def b(self):
        """Return b vector used in QP."""
        return self._b

    ###########
    # Methods #
    ###########

    def _update(self):
        """Update the task.

        Returns:
            np.array: A matrix used in QP.
            np.array: b vector used in QP.
        """
        pass

    def update(self):
        if self.tasks:
            for hard_task in self.tasks:
                results = [soft_task.update() for soft_task in hard_task]
                As = np.vstack([result[0] for result in results])
                bs = np.vstack([result[1] for result in results])
            # TODO: continue for hard priority tasks
            return As, bs
        return self._update()

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a string representing the class."""
        if self.tasks:
            tasks = []
            for i, soft_tasks in enumerate(self.tasks):
                results = []
                for task in soft_tasks:
                    if task.weight == 1:
                        results.append(str(task))
                    else:
                        results.append(str(task.weight) + ' * ' + str(task))
                soft_task = ' + '.join(results)
                tasks.append('Priority {}: '.format(i+1) + soft_task)
            return '\n'.join(tasks)
        return self.__class__.__name__

    def __str__(self):
        """Return a string describing the class."""
        return self.__repr__()

    def __call__(self):
        return self.update()

    def __add__(self, other):  # TODO: check when other has some tasks
        """Add a soft priority task."""
        if not isinstance(other, Task):
            raise TypeError("Expecting 'other' to be an instance of Task, instead got: {}".format(type(other)))
        if len(self.tasks) > 0:
            tasks = list(self.tasks)
            tasks[-1].append(other)
            return Task(tasks=tasks)
        return Task(tasks=[[self, other]])

    def __div__(self, other):
        """Add a hard priority task."""
        if not isinstance(other, Task):
            raise TypeError("Expecting 'other' to be an instance of Task, instead got: {}".format(type(other)))
        tasks = list(self.tasks)
        if other.tasks:
            # append all the other tasks
            for task in other.tasks:
                tasks.append(task)
        else:
            tasks.append([other])
        return Task(tasks=tasks)

    def __lshift__(self, other):
        """Insert a constraint (in-place operation)."""
        if not isinstance(other, Constraint):
            raise TypeError("Expecting 'other' to be an instance of Constraint, instead got: {}".format(type(other)))
        self._constraints.append(other)

    def __mul__(self, other):
        """Multiply the task by a relative weight."""
        if not isinstance(other, (int, float)) or other < 0:
            raise TypeError("Expecting a positive integer or float for the weight.")
        self.weight = other

    def __rmul__(self, other):
        self.__mul__(other)

    def __getitem__(self, key):
        """Get the corresponding task.

        Examples:
            >>> task1 = Task(weight=1)
            >>> task2 = Task(weight=2)
            >>> task = Task(tasks=[[task1, task2], [task1]])
            >>> task2 == task[0,1]  # get the second priority task in the first hard task, i.e. it will return task2
            True
        """
        pass


# Tests
if __name__ == '__main__':
    task1 = Task(weight=2)
    task2 = Task(weight=3)
    task = Task(tasks=[[task1, task2], [task1]])
    print(task)
