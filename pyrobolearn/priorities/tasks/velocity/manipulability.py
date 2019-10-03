# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the manipulability task.

The manipulability task implements a task that tries to maximize the manipulability measure given in [1]:

.. math:: w(q) = \sqrt{ \det( J(q) W J(q)^\top ) }

where :math:`W` is a constant weight matrix, :math:`q` are the joint positions, and :math:`J(q)` is the jacobian.
The gradient of :math:`w` is then computed and projected using the gradient projection method [2].

The quadratic cost being minimized is:

.. math:: ||\dot{q} - \dot{q}_0||^2

where :math:`\dot{q}` are the joint velocities being optimized,
:math:`\dot{q}_0 = k_0 \left( \frac{\partial w(q)}{\partial q} \right)^\top` where :math:`k_0 > 0` and
:math:`w(q)` is an objective function of the joint variables, where in this case, the manipulability measure is
given by :math:`w(q) = \sqrt{\det( J(q) J^\top(q) )}`. By maximizing this measure, we move away from singularities.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

# TODO: finish to implement this

import numpy as np

from pyrobolearn.priorities.tasks import JointVelocityTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ManipulabilityTask(JointVelocityTask):
    r"""Manipulability Task

    The manipulability task implements a task that tries to maximize the manipulability measure given in [1]:

    .. math:: w(q) = \sqrt{ \det( J(q) W J(q)^\top ) }

    where :math:`W` is a constant weight matrix, :math:`q` are the joint positions, and :math:`J(q)` is the jacobian.
    The gradient of :math:`w` is then computed and projected using the gradient projection method [2].

    The quadratic cost being minimized is:

    .. math:: ||\dot{q} - \dot{q}_0||^2

    where :math:`\dot{q}` are the joint velocities being optimized,
    :math:`\dot{q}_0 = k_0 \left( \frac{\partial w(q)}{\partial q} \right)^\top` where :math:`k_0 > 0` and
    :math:`w(q)` is an objective function of the joint variables, where in this case, the manipulability measure is
    given by :math:`w(q) = \sqrt{\det( J(q) J^\top(q) )}`. By maximizing this measure, we move away from singularities.

    References:
        - [1] "Robotics: Modelling, Planning, and Control", Siciliano et al., 2010
        - [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface
            weight (float, np.array[float[N,N]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(ManipulabilityTask, self).__init__(model=model, weight=weight, constraints=constraints)

        raise NotImplementedError("This class has not been implemented yet.")
