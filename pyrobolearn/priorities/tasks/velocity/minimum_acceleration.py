#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the minimum acceleration task.

The minimum acceleration task tries to minimize the change in velocity, that is, it minimizes:

.. math:: || \dot{q} - \dot{q}_t ||^2

where :math:`\dot{q}` are the joint velocities being optimized, and :math:`\dot{q}_t` are the current joint velocities.

This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A = I`, :math:`x = \dot{q}`,
and :math:`b = \dot{q}_t`.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

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


class MinAccelerationTask(JointVelocityTask):
    r"""Minimum Acceleration Task

    The minimum acceleration task tries to minimize the change in velocity, that is, it minimizes:

    .. math:: || \dot{q} - \dot{q}_t ||^2

    where :math:`\dot{q}` are the joint velocities being optimized, and :math:`\dot{q}_t` are the current joint
    velocities.

    This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A = I`,
    :math:`x = \dot{q}`, and :math:`b = \dot{q}_t`.


    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            weight (float, np.array[float[N,N]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        # the variables A and b are initialized by default to be A=I and b=0
        super(MinAccelerationTask, self).__init__(model=model, weight=weight, constraints=constraints)

    ###########
    # Methods #
    ###########

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        self._b = self.model.get_joint_velocities()
