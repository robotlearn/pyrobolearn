#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the force manipulability task.

 The manipulability task implements a tasks that tries to maximize the force manipulability measure given in [1]:

.. math:: w(q) = \sqrt( \det( (J(q) W J(q)^\top)^{-1} ) )

where :math:`W` is a constant weight matrix, :math:`q` are the joint positions, and :math:`J(q)` is the jacobian.
The gradient of :math:`w` is then computed and projected using the gradient projection method [2].
"""

import numpy as np

from pyrobolearn.priorities.tasks import ForceTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ForceManipulabilityTask(ForceTask):
    r"""Force Manipulability Task

    The manipulability task implements a tasks that tries to maximize the force manipulability measure given in [1]:

    .. math:: w(q) = \sqrt( \det( (J(q) W J(q)^\top)^{-1} ) )

    where :math:`W` is a constant weight matrix, :math:`q` are the joint positions, and :math:`J(q)` is the jacobian.
    The gradient of :math:`w` is then computed and projected using the gradient projection method [2].

    References:
        - [1] "Robotics: Modelling, Planning, and Control", Siciliano et al., 2010
        - [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            weight (float, np.array[float[6,6]], np.array[float[3,3]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(ForceManipulabilityTask, self).__init__(model=model, weight=weight, constraints=constraints)
