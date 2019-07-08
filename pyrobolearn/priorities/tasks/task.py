#!/usr/bin/env python
r"""Provide the various tasks (i.e. objective functions) used in QP.

The tasks presented here represents the quadratic objective functions used in quadratic programming (QP).

A quadratic program (QP) is written in standard form [1]_ as:

.. math::

    x^* =& \arg \min_x \; \frac{1}{2} x^T Q x + p^T x \\
    & \text{subj. to } \; \begin{array}{c} Gx \leq h \\ Fx = c \end{array}


where :math:`x` is the vector being optimized (in robotics, it can be joint positions, velocities, torques, ...),
"the matrix :math:`Q` and vector :math:`p` are used to define any quadratic objective function of these variables,
while the matrix-vector couples :math:`(G,h)` and :math:`(F,c)` respectively define inequality and equality
constraints" [1]_. Inequality constraints can include the lower bounds and upper bounds of :math:`x` by setting
:math:`G` to be the identity matrix or minus this one, and :math:`h` to be the upper or minus the lower bounds.

For instance, the quadratic objective function :math:`||Ax - b||_{W}^2` (where :math:`W` is a symmetric weight matrix)
is given in the standard form as:

.. math:: ||Ax - b||_{W}^2 = (Ax - b)^\top W (Ax - b) = x^\top A^\top W A x - 2 b^\top W A x + b^\top W b

where the last term :math:`b^\top W b` can be removed as it does not depend on the variables we are optimizing (i.e.
:math:`x`). We thus have :math:`Q = A^\top W A` a symmetric matrix and :math:`p = -2 A^\top W b`.

Note that if we had instead :math:`||Ax - b||_{W}^2 + c^\top x`, this could be rewritten as:

.. math:: ||Ax - b||_{W}^2 + c^\top x = x^\top A^\top W A x - (2 b^\top W A - c^\top) x + b^\top W b,

giving :math:`Q = A^\top W A` and :math:`p = (c - 2 A^\top W b)`.

Many control problems in robotics can be formulated as a quadratic programming problem. For instance, let's assume
that we want to optimize the joint velocities :math:`\dot{q}` given the end-effector's desired position and velocity
in task space. We can define the quadratic problem as:

.. math:: || J(q) \dot{q} - v_c ||^2

where :math:`v_c = K_p (x_d - x) + K_d (v_d - \dot{x})` (using PD control), with :math:`x_d` and :math:`x` the desired
and current end-effector's position respectively, and :math:`v_d` is the desired velocity. The solution to this
task (i.e. optimization problem) is the same solution given by `inverse kinematics`. Now, you can even obtain the
damped least squares inverse kinematics by adding a soft task such that
:math:`||J(q)\dot{q} - v_c||^2 + ||q||^2` is optimized (note that :math:`||q||^2 = ||A q - b||^2`, where :math:`A=I` is
the identity matrix and :math:`b=0` is the zero/null vector).


- **Soft** priority tasks: with soft-priority tasks, the quadratic programming problem being minimized for :math:`n`
  such tasks is given by:

  .. math::

      \begin{array}{c}
      x^* = \arg \min_x ||A_1 x - b_1||_{W_1}^2 + ||A_2 x - b_2 ||_{W_2}^2 + ... + ||A_n x - b_n ||_{W_n}^2 \\
      \text{subj. to } \; \begin{array}{c} Gx \leq h \\ Fx = c \end{array}
      \end{array}

  Often, the weight PSD matrices :math:`W_i` are just positive scalars :math:`w_i`. This problem can notably be solved
  by stacking the :math:`A_i` one of top of another, and stacking the :math:`b_i` and :math:`W_i` in the same manner,
  and solving :math:`||A x - b||_{W}^2`. This is known as the augmented task. When the matrices :math:`A_i` are
  Jacobians this is known as the augmented Jacobian (which can sometimes be ill-conditioned).

- **Hard** priority tasks: with hard-priority tasks, the quadratic programming problem for :math:`n` tasks is defined
  in a sequential manner, where the first most important task will be first optimized, and then the subsequent tasks
  will be optimized one after the other. Thus, the first task to be optimized is given by:

  .. math::

       x_1^* =& \arg \min_x \; ||A_1 x - b_1||^2 \\
       & \text{subj. to } \; \begin{array}{c} G_1 x \leq h_1 \\ F_1 x = c_1 \end{array}

  while the second next most important task that would be solved is given by:

  .. math::

      x_2^* =& \arg \min_x \; ||A_2 x - b_2||^2 \\
      &  \begin{array}{cc} \text{subj. to } & G_2 x \leq h_2 \\
                & F_2 x = c_2 \\
                & A_1 x = A_1 x_1^* \\
                & G_1 x \leq h_1 \\
                & F_1 x = c_1, \end{array}

  until the :math:`n` most important task, given by:

  .. math::

      x_n^* =& \arg \min_x  \; ||A_n x - b_n||^2 \\
      &  \begin{array}{cc} \text{subj. to } & A_1 x = A_1 x_1^* \\
                & ... \\
                & A_{n-1} x = A_{n-1} x_{n-1}^* \\
                & G_1 x \leq h_1 \\
                & ... \\
                & G_n x \leq h_n \\
                & F_1 x = c_1 \\
                & ... \\
                & F_n x = c_n. \end{array}

  By setting the previous :math:`A_{i-1} x = A_{i-1} x_{i-1}^*` as equality constraints, the current solution
  :math:`x_i^*` won't change the optimality of all higher priority tasks.

The implementation of this class and the subsequent classes is inspired by [2] (which is licensed under the LGPLv2).

References:
    - [1] "Quadratic Programming in Python" (https://scaron.info/blog/quadratic-programming-in-python.html), Caron, 2017
    - [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [3] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
"""

import numpy as np

from pyrobolearn.priorities.models import ModelInterface
from pyrobolearn.priorities.constraints.constraint import Constraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: take into account constraints
# TODO: take into account hard priority tasks

class Task(object):
    r"""Task (abstract) class.

    This class describes the Task or Stack of Tasks (SoT).

    Python implementation of Tasks based on the OpenSoT framework [1].

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN"
            `code <https://opensot.wixsite.com/opensot>`_,
            `slides <https://docs.google.com/presentation/d/1kwJsAnVi_3ADtqFSTP8wq3JOGLcvDV_ypcEEjPHnCEA>`_,
            `tutorial video <https://www.youtube.com/watch?v=yFon-ZDdSyg>`_,
            `old code <https://github.com/songcheng/OpenSoT>`_), Rocchi et al., 2015
    """

    def __init__(self, stack_of_tasks=[], model=None, weight=1., constraints=[]):
        """
        Initialize the task, or stack of tasks.

        Args:
            stack_of_tasks (list of list of Task, empty list): stack of tasks represented as list of list of tasks,
                where the list is ordered by hard priorities, and the nested list contains soft priority tasks that
                have to be weighted together.
            model (ModelInterface, None): model interface associated to the task.
            weight (float, np.array[N,N]): weight scalar or matrix associated to the task.
            constraints (list of Constraint): list of constraints associated to the task.
        """
        # set and check each given parameter
        self.tasks = stack_of_tasks
        self.model = model
        self.weight = weight
        self.constraints = constraints

        # check that the task is a valid task or a stack o tasks
        if self.model is None and not self.is_stack_of_tasks():
            raise RuntimeError("Expecting the task to be a valid task or a stack of tasks. You can not instantiate "
                               "an empty task.")

        # set the number of variables to optimize
        self._x_size = self.model.num_dofs

        # define task matrix and vector
        if self.model is not None and not self.is_stack_of_tasks():
            self._A = np.identity(self.x_size)  # None
            self._b = np.zeros(self.x_size)  # None
            self._c = np.zeros(self.x_size)  # None

    ##############
    # Properties #
    ##############

    @property
    def tasks(self):
        """Return the tasks."""
        return self._tasks

    @tasks.setter
    def tasks(self, tasks):
        """Set the stack of tasks."""
        # check type
        if tasks is None:
            tasks = []
        if not isinstance(tasks, (list, tuple, set)):
            raise TypeError("Expecting the given 'tasks', to be a list of list of `Task`, instead got: "
                            "{}".format(type(tasks)))

        # go through the stack of tasks
        for i, hard_task in enumerate(tasks):

            # if the hard task is a list of soft tasks
            if isinstance(hard_task, (list, tuple, set)):
                # go through each soft task in the hard task
                for j, soft_task in enumerate(hard_task):
                    if not isinstance(soft_task, Task):
                        raise TypeError("The given task positioned at ({}, {}) is not an instance of `Task`, but: "
                                        "{}".format(i, j, type(soft_task)))
            else:   # if not, check that the hard task is an instance of Task
                if not isinstance(hard_task, Task):
                    raise TypeError("Expecting the {}th hard task to be an instance of `Task` or a list of `Task`, "
                                    "instead got: {}".format(i, type(hard_task)))

        # set the stack of tasks
        self._tasks = tasks

    @property
    def model(self):
        """Return the model interface."""
        return self._model

    @model.setter
    def model(self, model):
        """Set the model interface."""
        if model is not None and not isinstance(model, ModelInterface):
            raise TypeError("Expecting the given 'model' to be None or an instance of `ModelInterface`, instead got: "
                            "{}".format(type(model)))
        self._model = model

    @property
    def weight(self):
        """Return the relative weight (used for soft priorities)."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        """Set the weight scalar or matrix."""
        # check the type
        if not isinstance(weight, (float, int, np.ndarray)):
            raise TypeError("Expecting the given 'weight' to be an int, float, or np.ndarray, instead got: "
                            "{}".format(type(weight)))

        # if weight is a matrix, check that it is PSD
        if isinstance(weight, np.ndarray):
            if not self._is_positive_semidefinite(weight):
                raise ValueError("Expecting the relative weight matrix to be positive semidefinite (PSD).")

        # if weight is a scalar, check it is positive
        else:
            if weight < 0:
                raise ValueError("Expecting the relative weight to be positive.")

        # set the weight
        self._weight = weight

    @property
    def depth(self):
        """Return the depth of the stack of tasks."""
        return len(self.tasks)

    @property
    def constraints(self):
        """Return the constraints."""
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        """Set the constraints."""
        # check the type
        if constraints is None:
            constraints = []
        if not isinstance(constraints, (list, tuple, set)):
            constraints = [constraints]

        # go through each constraint and check its type
        for i, constraint in enumerate(constraints):
            if not isinstance(constraint, Constraint):
                raise TypeError("The {}th given constraint is not an instance of `Constraint`, instead got: "
                                "{}".format(i, type(constraint)))

        # set the constraints associated with the task
        self._constraints = constraints

    @property
    def x_size(self):
        """Return the number of variables being optimized."""
        return self._x_size

    @property
    def num_tasks(self):
        """Return the total number of tasks defined."""
        return self.get_num_tasks()

    @property
    def num_hard_tasks(self):
        """Return the number of hard tasks. This counts the task itself if the instance is not a stack of tasks."""
        return self.get_num_hard_tasks()

    @property
    def A(self):
        r"""Return the A matrix from :math:`||Ax - b||^2` used in QP.

        Returns:
        """
        if self.is_stack_of_tasks():
            pass
        return self._A

    @property
    def b(self):
        r"""Return the b vector from :math:`||Ax - b||^2` used in QP."""
        if self.is_stack_of_tasks():
            pass
        return self._b

    @property
    def c(self):
        r"""Return the c vector from :math:`||Ax - b||^2 + c^\top x` used in QP."""
        if self.is_stack_of_tasks():
            pass
        return self._c

    @property
    def Q(self):
        r"""Return the Q matrix :math:`Q = A^\top W A` used in :math:`\frac{1}{2} x^T Q x + p^T x` for QP."""
        if self.is_stack_of_tasks():
            pass
        return self._A.T.dot(self.weight).dot(self._A)

    @property
    def p(self):
        r"""Return the p vector :math:`p = (c - 2 A^\top W b)` used in :math:`\frac{1}{2} x^T Q x + p^T x`
        for QP."""
        if self.is_stack_of_tasks():
            pass
        return self.c - self._A.T.dot(self.weight).dot(self._b)

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def _is_positive_semidefinite(x, tol=1e-8):
        """Check if the given argument is a PSD matrix.

        Args:
            x (np.array): matrix to check if it a PSD matrix.
            tol (float): tolerance.
        """
        return np.all(np.linalg.eigvals(x) >= tol)

    def _check_weight_matrix_shape(self, shape):
        """
        Check if the weight matrix has the correct shape.

        Args:
            shape (tuple of int): shape that the weight matrix should have.

        Raises:
            ValueError: if the shape is not the correct one.
        """
        if isinstance(self.weight, np.ndarray) and self.weight.shape != shape:
            raise ValueError("Expecting the given weight matrix to have a shape of {}, but instead got a shape of: "
                             "{}".format(shape, self.weight.shape))

    ###########
    # Methods #
    ###########

    def is_stack_of_tasks(self):
        """Check if the task is a stack of tasks. This returns True even if there is one task in the stack of tasks."""
        return len(self.tasks[0]) > 0

    def is_soft_task(self):
        """Check if the task is a soft task."""
        return len(self.tasks) == 1 and len(self.tasks[0]) > 1

    def get_num_tasks(self):
        """Return the total number of tasks."""
        # create counter
        cnt = 0

        # go through the stack of tasks and increment the counter for each encountered task
        for hard_task in self.tasks:
            for soft_task in hard_task:
                cnt += 1

        # if there was nothing in the stack of tasks, set counter to 1 (because the instance is then a task)
        if cnt == 0:
            cnt = 1

        return cnt

    def get_num_hard_tasks(self):
        """
        Return the number of hard tasks (i.e. the number of levels) in the stack of tasks. This counts the task itself
        if the instance is not a stack of tasks.

        Returns:
            int: number of hard tasks (between 1 and `len(self.tasks)`)
        """
        return len(self.tasks)

    def get_num_soft_tasks(self, level):
        """
        Return the number of soft tasks there are at the specified level in the stack of tasks.

        Args:
            level (int): level in the stack of tasks which is in [0, ..., len(self.tasks)].

        Returns:
            int: the number of soft tasks
        """
        return len(self.tasks[level])

    def get_soft_tasks(self, hard_task_idx=0, soft_task_idx=None):
        r"""
        Return the specified soft tasks.

        Args:
            hard_task_idx (int): level in the stack of tasks which is in [0, ..., len(self.tasks)]
            soft_task_idx (int): soft task index in `self.tasks[hard_task_idx]`. If None, it will return all the soft
                tasks present at the specified level `hard_task_idx`.

        Returns:
            Task, list of Task: the specified soft tasks.
        """
        if soft_task_idx is None:
            return self.tasks[hard_task_idx]
        else:
            return self.tasks[hard_task_idx][soft_task_idx]

    def loss(self, x):
        """
        Compute the error loss of the given task.

        Args:
            x (np.array): joint velocities that are being optimized.

        Returns:
            float: loss value
        """
        return np.sum((self._A.dot(x) - self._b) ** 2)

    def _update(self):
        """Update the task.

        Compute the A matrix and b vector that will be used by the task solver. This has to be implemented in the
        child classes.

        Returns:
            np.array: A matrix used in QP.
            np.array: b vector used in QP.
        """
        pass

    def update(self):
        """
        Compute the A matrix and b vector that will be used by the task solver.

        Returns:
            np.array: A matrix used in QP.
            np.array: b vector used in QP.
        """
        # if stack of tasks
        if self.tasks:
            for hard_task in self.tasks:
                results = [soft_task.update() for soft_task in hard_task]
                As = np.vstack([result[0] for result in results])
                bs = np.vstack([result[1] for result in results])
            # TODO: continue for hard priority tasks
            return As, bs

        # if normal task
        # update the constraints

        return self._update()

    #############
    # Operators #
    #############

    # def __repr__(self):
    #     """Return a string representing the class."""
    #     return self.__str__()

    def __str__(self):
        """Return a string describing the class."""
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

    def __call__(self):
        """Update the tasks."""
        return self.update()

    def __add__(self, other):  # TODO: check when other has some tasks
        """Add a soft priority task.

        Examples:
            task1 = Task(weight=2)
            task2 = Task(weight=3)
            task = task1 + task2
            print(task)
        """
        if not isinstance(other, Task):
            raise TypeError("Expecting 'other' to be an instance of Task, instead got: {}".format(type(other)))
        if len(self.tasks) > 0:
            tasks = list(self.tasks)
            tasks[-1].append(other)
            return Task(stack_of_tasks=tasks)
        return Task(stack_of_tasks=[[self, other]])

    def __div__(self, other):
        """Append a hard priority task to the stack of tasks.

        Examples:
            task1 = Task(weight=2)
            task2 = Task(weight=3)
            task = task1 / task2
            print(task)
        """
        # check type of
        if not isinstance(other, Task):
            raise TypeError("Expecting 'other' to be an instance of Task, instead got: {}".format(type(other)))
        tasks = list(self.tasks)
        if other.tasks:
            # append all the other tasks
            for task in other.tasks:
                tasks.append(task)
        else:
            tasks.append([other])
        return Task(stack_of_tasks=tasks)

    def __lshift__(self, other):
        """Insert a constraint (in-place operation).

        Examples:
            task1 = Task()
            task2 = Task()
            task = task1 + task2
            constraint = Constraint()

            task << constraint
            print(task1.constraint)
            print(task2.constraint)
        """
        # check other type; it must be a constraint
        if not isinstance(other, Constraint):
            raise TypeError("Expecting 'other' to be an instance of Constraint, instead got: {}".format(type(other)))

        # if we have a stack of tasks, insert the constraint for all tasks
        if self.tasks:
            for hard_task in self.tasks:
                if isinstance(hard_task, list):
                    for soft_task in hard_task:
                        soft_task << other
                else:
                    hard_task << other

        # if we have one task, append the constraint
        else:
            self.constraints.append(other)

    def __mul__(self, other):
        """Multiply the task by a relative weight scalar or matrix."""
        self.weight = other

    def __rmul__(self, other):
        """Multiply the task by a relative weight"""
        return self.__mul__(other)

    def __getitem__(self, key):
        """Get the corresponding task.

        Args:
            key (int, slice, tuple of int): key.

        Examples:
            # >>> task1 = Task(weight=1)
            # >>> task2 = Task(weight=2)
            # >>> task = Task(stack_of_tasks=[[task1, task2], [task1]])
            # >>> task2 == task[0,1]  # get the second priority task in the first hard task, i.e. it will return task2
            # True
        """
        if isinstance(key, tuple) and len(key) == 2:
            return self.tasks[key[0]][key[1]]
        return self.tasks[key]


class KinematicTask(Task):
    r"""Kinematic Task

    Kinematic tasks focus on tasks that optimize joint velocities.
    """
    pass


class JointVelocityTask(Task):
    r"""Joint Velocity Task

    Joint velocity tasks are tasks that optimize joint velocities :math:`\dot{q}`.
    """
    pass


class DynamicTask(Task):
    r"""Dynamic Task

    Dynamic tasks focus on tasks that involve accelerations and forces / torques.
    """
    pass


class JointAccelerationTask(Task):
    r"""Joint Acceleration Task

    Joint acceleration tasks are tasks that optimize joint accelerations :math:`\ddot{q}`.
    """
    pass


class JointTorqueTask(Task):
    r"""Joint Torque Task

    Joint torque tasks are tasks that optimize joint torques :math:`\tau`.
    """
    pass


# Tests
if __name__ == '__main__':
    task1 = Task(weight=2)
    task2 = Task(weight=3)
    task = Task(stack_of_tasks=[[task1, task2], [task1]])
    print(task)

    print(task2 == task[0, 1])

    task = 1./2 * task1 + 1./3 * task2
    task = task / task1

    print(task)
