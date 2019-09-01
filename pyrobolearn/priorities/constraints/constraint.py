#!/usr/bin/env python
r"""Provide the various constraints and bounds used in QP.

Provide the various optimization constraints (:math:`G, h, F, c` in the upcoming formulation) used in QP.

A quadratic program (QP) is written in standard form [1]_ as:

.. math::

    x^* =& \arg \min_x \; \frac{1}{2} x^T Q x + p^T x \\
    & \text{subj. to } \; \begin{array}{c} Gx \leq h \\ Fx = k \end{array}


where :math:`x` is the vector being optimized (in robotics, it can be joint positions, velocities, torques, ...),
"the matrix :math:`Q` and vector :math:`p` are used to define any quadratic objective function of these variables,
while the matrix-vector couples :math:`(G,h)` and :math:`(F,k)` respectively define inequality and equality
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
      \text{subj. to } \; \begin{array}{c} Gx \leq h \\ Fx = k \end{array}
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
       & \text{subj. to } \; \begin{array}{c} G_1 x \leq h_1 \\ F_1 x = k_1 \end{array}

  while the second next most important task that would be solved is given by:

  .. math::

      x_2^* =& \arg \min_x \; ||A_2 x - b_2||^2 \\
      &  \begin{array}{cc} \text{subj. to }
                & G_2 x \leq h_2 \\
                & F_2 x = k_2 \\
                & A_1 x = A_1 x_1^* \\
                & G_1 x \leq h_1 \\
                & F_1 x = k_1,
        \end{array}

  until the :math:`n` most important task, given by:

  .. math::

      x_n^* =& \arg \min_x  \; ||A_n x - b_n||^2 \\
      &  \begin{array}{cc} \text{subj. to } & A_1 x = A_1 x_1^* \\
                & ... \\
                & A_{n-1} x = A_{n-1} x_{n-1}^* \\
                & G_1 x \leq h_1 \\
                & ... \\
                & G_n x \leq h_n \\
                & F_1 x = k_1 \\
                & ... \\
                & F_n x = k_n. \end{array}

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


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Constraint(object):
    r"""Constraint (abstract) class.

    Constraints can be classified into 2 groups:

    1. inequality constraints
        - bounds: math:`lb \leq x \leq ub`.
        - bilateral: :math:`b_l \leq A_{ineq} x \leq b_u`
        - unilateral: math:`b_l \leq A_{ineq} x`, or :math:`A_{ineq} x \leq b_u`
    2. equality constraints: math:`A_{eq} = b_{eq}`.

    In the robotics case, they can also be divided into 2 groups on another axis:

    1. kinematics constraints: take only into account kinematic information such as velocities.
    2. dynamic constraints: take into account forces, accelerations, and inertia.


    When using Quadratic programming, we only consider linear constraints. Non-linear constraints thus have to be
    linearized in order to be used. This is for instance the case with a friction cone (which provides a non-linear
    constraint) and its linearization; the friction pyramid.


    Python implementation of Constraints based on the OpenSoT framework [1].

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN"
            ([code](https://opensot.wixsite.com/opensot),
            [slides](https://docs.google.com/presentation/d/1kwJsAnVi_3ADtqFSTP8wq3JOGLcvDV_ypcEEjPHnCEA),
            [tutorial video](https://www.youtube.com/watch?v=yFon-ZDdSyg),
            [old code](https://github.com/songcheng/OpenSoT), LGPLv2), Rocchi et al., 2015
    """

    def __init__(self, model=None, constraints=[]):
        """
        Initialize the Constraint.

        Args:
            model (ModelInterface): robotic model interface.
            constraints (list[Constraint], None): inner constraints. By providing a list of constraints, they can be
              combined easily.
        """
        self.constraints = constraints
        self.model = model

        # variables to be set in the corresponding child classes
        self._lower_bound = None
        self._upper_bound = None
        self._A_eq = None
        self._b_eq = None
        self._A_ineq = None
        self._b_lower_bound = None
        self._b_upper_bound = None

    ##############
    # Properties #
    ##############

    @property
    def model(self):
        """Return the model interface."""
        return self._model

    @model.setter
    def model(self, model):
        """Set the model interface."""
        if model is not None and not isinstance(model, ModelInterface):
            raise TypeError("Expecting the given 'model' to be an instance of `ModelInterface`, instead got: "
                            "{}".format(model))
        self._model = model

    @property
    def x_size(self):
        """Return the number of variables being optimized."""
        # return self._x_size
        if self.model is not None:
            return self.model.num_actuated_joints
        return 0

    @property
    def constraints(self):
        """Return the dict of inner constraints."""
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        if constraints is None:
            constraints = dict()
        elif isinstance(constraints, dict):
            equality_constraints = constraints.get(EqualityConstraint, [])
            inequality_constraints = constraints.get(InequalityConstraint, [])
            constraints = equality_constraints + inequality_constraints
        elif not isinstance(constraints, (list, tuple)):
            constraints = [constraints]

        constraint_dict = dict()
        for i, constraint in enumerate(constraints):
            if isinstance(constraint, EqualityConstraint):
                constraint_dict.setdefault(EqualityConstraint, []).append(constraint)
            elif isinstance(constraint, InequalityConstraint):
                constraint_dict.setdefault(InequalityConstraint, []).append(constraint)
                if isinstance(constraint, BoundConstraint):
                    constraint_dict.setdefault(BoundConstraint, []).append(constraint)
                elif isinstance(constraint, UnilateralConstraint):
                    constraint_dict.setdefault(UnilateralConstraint, []).append(constraint)
                    if isinstance(constraint, LowerUnilateralConstraint):
                        constraint_dict.setdefault(LowerUnilateralConstraint, []).append(constraint)
                    elif isinstance(constraint, UpperUnilateralConstraint):
                        constraint_dict.setdefault(UpperUnilateralConstraint, []).append(constraint)
                elif isinstance(constraint, BilateralConstraint):
                    constraint_dict.setdefault(BilateralConstraint, []).append(constraint)
                else:
                    raise TypeError("Expecting the {}th inequality constraint to be an instance of `BoundConstraint`,"
                                    " `UnilateralConstraint`, `BilateralConstraint`, but got: "
                                    "{}".format(i, type(constraint)))
            else:
                raise TypeError("Expecting the {}th constraint to be an instance of `EqualityConstraint` or "
                                "`InequalityConstraint`, but got: {}".format(i, type(constraint)))

        self._constraints = constraint_dict

    @property
    def lower_bound(self):
        r"""Return the lower bound of the optimization variables: :math:`b_l \leq x`.

        Returns:
            np.array[float[N]]: lower bound.
        """
        if self.constraints:
            constraints = self.constraints.get(BoundConstraint, [])
            return [constraint.lower_bound for constraint in constraints]
        return self._lower_bound

    @property
    def upper_bound(self):
        r"""Return the upper bound of the optimization variables: :math:`x \leq b_u`.

        Returns:
            np.array[float[N]]: upper bound.
        """
        if self.constraints:
            constraints = self.constraints.get(BoundConstraint, [])
            return [constraint.lower_bound for constraint in constraints]
        return self._upper_bound

    @property
    def A_eq(self):
        r"""Return the equality constraint matrix :math:`A_{eq}`, such that :math:`A_{eq} x = b_{eq}`.

        Returns:
            np.array[float[N,N]]: equality constraint matrix.
        """
        if self.constraints:
            constraints = self.constraints.get(EqualityConstraint, [])
            return [constraint.A_eq for constraint in constraints]
        return self._A_eq

    @property
    def b_eq(self):
        r"""Return the equality constraint vector :math:`b_{eq}`, such that :math:`A_{eq} x = b_{eq}`.

        Returns:
            np.array[float[N]]: equality constraint vector.
        """
        if self.constraints:
            constraints = self.constraints.get(EqualityConstraint, [])
            return [constraint.b_eq for constraint in constraints]
        return self._b_eq

    @property
    def A_ineq(self):
        r"""Return the inequality constraint matrix :math:`A_{ineq}`, such that :math:`b_l \leq A_{ineq} x \leq b_u`.

        Returns:
            np.array[float[N,N]]: inequality constraint matrix.
        """
        if self.constraints:
            unilateral_constraints = self.constraints.get(UnilateralConstraint, [])
            bilateral_constraints = self.constraints.get(BilateralConstraint, [])
            return [constraint.A_ineq for constraint in unilateral_constraints + bilateral_constraints]
        return self._A_ineq

    @property
    def b_lower_bound(self):
        r"""Return the lower bound of the inequality constraint: :math:`b_l \leq A_{ineq} x`.

        Returns:
            np.array[float[N]]: inequality constraint lower bound vector.
        """
        if self.constraints:
            unilateral_constraints = self.constraints.get(LowerUnilateralConstraint, [])
            bilateral_constraints = self.constraints.get(BilateralConstraint, [])
            return [constraint.b_lower_bound for constraint in unilateral_constraints + bilateral_constraints]
        return self._b_lower_bound

    @property
    def b_upper_bound(self):
        r"""Return the upper bound of the inequality constraint: :math:`A_{ineq} x \leq b_u`.

        Returns:
            np.array[float[N]]: inequality constraint upper bound vector.
        """
        if self.constraints:
            unilateral_constraints = self.constraints.get(UpperUnilateralConstraint, [])
            bilateral_constraints = self.constraints.get(BilateralConstraint, [])
            return [constraint.b_upper_bound for constraint in unilateral_constraints + bilateral_constraints]
        return self._b_upper_bound

    @property
    def G(self):
        r"""Return the inequality constraint matrix :math:`G` used in inequality constraints :math:`Gx \leq h` in QP.

        Returns:
            np.array[float[N,N]]: inequality constraint matrix.
        """
        if self.constraints:
            results = []
            bound_constraints = self.constraints.get(BoundConstraint, [])
            for constraint in bound_constraints:
                x_size = len(constraint.lower_bound)
                results.append(-np.identity(x_size))
                results.append(np.identity(x_size))
            bilateral_constraints = self.constraints.get(BilateralConstraint, [])
            for constraint in bilateral_constraints:
                results.append(-constraint.A_ineq)
                results.append(constraint.A_ineq)
            lower_unilateral_constraints = self.constraints.get(LowerUnilateralConstraint, [])
            for constraint in lower_unilateral_constraints:
                results.append(-constraint.A_ineq)
            upper_unilateral_constraints = self.constraints.get(UpperUnilateralConstraint, [])
            for constraint in upper_unilateral_constraints:
                results.append(constraint.A_ineq)

            if results:
                return np.concatenate(results)
        else:
            if isinstance(self, BoundConstraint):
                x_size = len(self._lower_bound)
                return np.concatenate((-np.identity(x_size), np.identity(x_size)))
            elif isinstance(self, BilateralConstraint):
                return np.concatenate((-self._A_ineq, self._A_ineq))
            elif isinstance(self, LowerUnilateralConstraint):
                return -self._A_ineq
            elif isinstance(self, UpperUnilateralConstraint):
                return self._A_ineq

    @property
    def h(self):
        r"""Return the inequality constraint vector :math:`h` used in inequality constraints :math:`Gx \leq h` in QP.

        Returns:
            np.array[float[N]]: inequality constraint vector.
        """
        if self.constraints:
            results = []
            bound_constraints = self.constraints.get(BoundConstraint, [])
            for constraint in bound_constraints:
                results.append(-constraint.lower_bound)
                results.append(constraint.upper_bound)
            bilateral_constraints = self.constraints.get(BilateralConstraint, [])
            for constraint in bilateral_constraints:
                results.append(-constraint.b_lower_bound)
                results.append(constraint.b_upper_bound)
            lower_unilateral_constraints = self.constraints.get(LowerUnilateralConstraint, [])
            for constraint in lower_unilateral_constraints:
                results.append(-constraint.b_lower_bound)
            upper_unilateral_constraints = self.constraints.get(UpperUnilateralConstraint, [])
            for constraint in upper_unilateral_constraints:
                results.append(constraint.b_upper_bound)

            if results:
                return np.concatenate(results)
        else:
            if isinstance(self, BoundConstraint):
                return np.concatenate((-self._lower_bound, self._upper_bound))
            elif isinstance(self, BilateralConstraint):
                return np.concatenate((-self._b_lower_bound, self._upper_bound))
            elif isinstance(self, LowerUnilateralConstraint):
                return -self._b_lower_bound
            elif isinstance(self, UpperUnilateralConstraint):
                return self._b_upper_bound

    @property
    def F(self):
        r"""Return the equality constraint matrix :math:`F` used in equality constraints :math:`Fx = k` in QP.

        Returns:
            np.array[float[N,N]]: equality constraint matrix.
        """
        return self.A_eq

    @property
    def k(self):
        r"""Return the equality constraint vector :math:`c` used in equality constraints :math:`Fx = k` in QP.

        Returns:
            np.array[float[N]]: equality constraint vector.
        """
        return self.b_eq

    ##################
    # Static methods #
    ##################

    # @staticmethod
    # def is_equality_constraint():
    #     r"""Return True if it is an equality constraint: :math:`A_{eq} x = b_{eq}`."""
    #     return False
    #
    # @staticmethod
    # def is_inequality_constraint():
    #     r"""Return True if it is an inequality constraint: :math:`b_l \leq A_{ineq} x \leq b_u`."""
    #     return False
    #
    # @staticmethod
    # def has_bounds():
    #     r"""Return True if it is a bound constraint: math:`b_l \leq x \leq b_u`."""
    #     return False
    #
    # @staticmethod
    # def is_unilateral_constraint():
    #     r"""Return True if it is a unilateral constraint: math:`b_l \leq A_{ineq} x` xor :math:`A_{ineq} x
    #     \leq b_u`."""
    #     return False
    #
    # @staticmethod
    # def is_bilateral_constraint():
    #     r"""Return True if it is a bilateral constraint: :math:`b_l \leq A_{ineq} x \leq b_u`."""
    #     return False

    ###########
    # Methods #
    ###########

    def has_constraints(self):
        """Return True if it has inner constraints."""
        return len(self.constraints) > 0

    def is_single_constraint(self):
        """Return True if single constraint."""
        return len(self.constraints) == 0

    def append(self, constraint):
        r"""
        Append a constraint to the list of inner constraints.

        Args:
            constraint (Constraint): constraint to append to the list.
        """
        # check type
        if not isinstance(constraint, Constraint):
            raise TypeError("Expecting the given 'constraint' to be an instance of `Constraint` but got instead: "
                            "{}".format(type(constraint)))
        if not constraint.is_single_constraint():
            raise ValueError("Expecting to append a single constraint, but the given constraint as a list of "
                             "inner constraints")

        if isinstance(constraint, EqualityConstraint):
            self.constraints.setdefault(EqualityConstraint, []).append(constraint)
        elif isinstance(constraint, InequalityConstraint):
            self.constraints.setdefault(InequalityConstraint, dict())
            if isinstance(constraint, BoundConstraint):
                self.constraints.setdefault(BoundConstraint, []).append(constraint)
            elif isinstance(constraint, UnilateralConstraint):
                self.constraints.setdefault(UnilateralConstraint, []).append(constraint)
                if isinstance(constraint, LowerUnilateralConstraint):
                    self.constraints.setdefault(LowerUnilateralConstraint, []).append(constraint)
                elif isinstance(constraint, UpperUnilateralConstraint):
                    self.constraints.setdefault(UpperUnilateralConstraint, []).append(constraint)
            elif isinstance(constraint, BilateralConstraint):
                self.constraints.setdefault(BilateralConstraint, []).append(constraint)
            else:
                raise TypeError("Expecting the inequality constraint to be an instance of `BoundConstraint`, "
                                "`UnilateralConstraint`, `BilateralConstraint`, but got: {}".format(type(constraint)))
        else:
            raise TypeError("The given type of constraint is not currently supported.")

    def _update(self):
        """Update the constraint variables. This has to be implemented in the child classes."""
        pass

    def update(self):
        r"""
        Update the various constraint matrices and vectors: :math:`A_{eq}, b_{eq}, A_{ineq}, b_l, b_u, ...`.
        """
        if self.constraints:
            for constraint in self.constraints:
                constraint.update()
        else:
            self._update()

    #############
    # Operators #
    #############

    # def __repr__(self):
    #     return self.__class__.__name__

    def __str__(self):
        """Return a string describing the class."""
        return self.__class__.__name__

    def __call__(self):
        """
        Update the constraint (i.e. update the various constraint matrices and vectors).
        """
        return self.update()

    def __len__(self):
        """
        Return the number of inner constraints.

        Returns:
            int: number of inner constraints.
        """
        if self.constraints:
            return len(self.constraints)
        return 1

    def __add__(self, other):
        """
        Add a constraint to the list of inner constraints.

        Args:
            other (Constraint): the other constraint.

        Returns:
            Constraint: the combined constraints.
        """
        # check other type
        if not isinstance(other, Constraint):
            raise TypeError("Expecting 'other' to be an instance of `Constraint` but got instead: "
                            "{}".format(type(other)))

        if self.constraints:
            constraints = dict(self.constraints)  # copy inner constraints
            if other.constraints:
                for key, value in other.constraints.items():
                    if isinstance(key, (EqualityConstraint, BoundConstraint, UnilateralConstraint,
                                        BilateralConstraint)):
                        constraints.setdefault(key, []).append(value)

        # TODO: finish to implement this

        return Constraint()


class NullConstraint(Constraint):
    r"""Null constraint.

    This is a dummy constraint which is used to specify that no constraints are used.
    """
    pass


class EqualityConstraint(Constraint):
    r"""Equality constraint."""
    pass


class InequalityConstraint(Constraint):
    r"""Inequality constraint"""
    pass


class UnilateralConstraint(InequalityConstraint):
    r"""Unilateral inequality constraint"""
    pass


class LowerUnilateralConstraint(UnilateralConstraint):
    r"""Lower unilateral inequality constraint."""
    pass


class UpperUnilateralConstraint(UnilateralConstraint):
    r"""Upper unilateral inequality constraint."""
    pass


class BilateralConstraint(InequalityConstraint):
    r"""Bilateral inequality constraint."""
    pass


class BoundConstraint(InequalityConstraint):
    r"""Bound inequality constraint."""
    pass


class KinematicConstraint(Constraint):
    r"""Kinematic constraint."""
    pass


class JointVelocityConstraint(KinematicConstraint):
    r"""Joint velocity constraint."""
    pass


class DynamicConstraint(Constraint):
    r"""Dynamic constraint."""
    pass


class JointAccelerationConstraint(DynamicConstraint):
    r"""Joint acceleration constraint."""
    pass


class JointEffortConstraint(DynamicConstraint):
    r"""Joint effort constraint."""
    pass


class JointTorqueConstraint(DynamicConstraint):
    r"""Joint torque constraint."""
    pass


class JointForceConstraint(DynamicConstraint):
    r"""Joint Force constraint."""
    pass
