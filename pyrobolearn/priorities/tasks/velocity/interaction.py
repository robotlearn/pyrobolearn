#!/usr/bin/env python
r"""Provide the interaction task.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

# TODO: finish to implement this class

import numpy as np

from pyrobolearn.priorities.tasks import JointVelocityTask
from pyrobolearn.priorities.tasks.velocity import CartesianTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman, Alessio Rocchi) (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class InteractionTask(JointVelocityTask):
    r"""Interaction Task

    From the documentation of the framework of [1], "The Interaction class implements an Admittance based force
    control using the admittance law:

    .. math::

        dx = K_p * (w_d - w) \\
        x_d = x + dx

    where :math:`w_d \in \mathbb{R}^6` is the desired wrench in some base_link frame, :math:`w` is the measured wrench
    transformed from the Force/Torque sensor frame to the base_link frame. The displacement :math:`dx` is integrated
    using the previous position :math:`x`, and a new desired position :math:`x_d` is computed. The references
    :math:`x_d` and :math:`dx` are then used inside a Cartesian task (see ``CartesianTask``).

    Warnings: the :math:`w_d` is the desired wrench that the robot has to exert on the environment, so the measured
    wrench :math:`w` is the wrench produced by the robot on the environment (and not the opposite)!"


    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, distal_link, base_link=-1, desired_wrench=0., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            distal_link (int, str): distal link id or name.
            base_link (int, str, None): base link id or name. If None, it will be the base root link.
            desired_wrench (float, np.array[float[6]]): desired wrench.
            weight (float, np.array[float[6,6]]): weight scalar or matrix associated to the task.
            constraints (list of Constraint): list of constraints associated with the task.
        """
        super(InteractionTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # create sub-task
        self._task = CartesianTask(model, distal_link=distal_link, base_link=base_link, weight=weight)

        raise NotImplementedError("This class has not been implemented yet.")

    def _update(self):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        self._A = self._task.A
        self._b = self._task.b
