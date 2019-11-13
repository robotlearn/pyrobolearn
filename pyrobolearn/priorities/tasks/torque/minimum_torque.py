#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the minimum torque task.

The minimum torque task minimizes the joint torques, that is it minimizes:

.. math:: ||\tau||^2,

which is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=I`, :math:`x=\tau`,
and :math:`b=0`.
"""

from pyrobolearn.priorities.tasks import JointTorqueTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MinTorqueTask(JointTorqueTask):
    r"""Minimum Torque Task

    The minimum torque task minimizes the joint torques, that is it minimizes:

    .. math:: ||\tau||^2,

    which is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=I`, :math:`x=\tau`,
    and :math:`b=0`.
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
        super(MinTorqueTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # first update
        self.update()
