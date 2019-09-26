#!/usr/bin/env python
r"""Provide the pure rolling (no sliding) task.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] "On the Kinematics of Wheeled Motion Control of a Hybrid Wheeled-Legged CENTAURO Robot", Kamedula et al., 2019
"""

# TODO: finish to implement this

import numpy as np

from pyrobolearn.priorities.tasks import JointVelocityTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Arturo Laurenzi (C++)", "Malgorzata Kamedula (insight)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PureRollingTask(JointVelocityTask):
    r"""Pure rolling (no-sliding) Task

    The pure rolling without slipping constrained the end-effector contact point (with the ground) to remain fixed:

    .. math:: v_{cp} = 0

    where :math:`v_{cp}` is the contact point velocity given by :math:`v_{cp} = \dot{x}_{cp}` where :math:`x_{cp}` is
    the position vector (from the world frame origin) to the contact point.

    The optimization problem can be formulated as:

    .. math:: || S J_c(q) \dot{q} ||^2

    where :math:`S = [0_{3 \times 3}, 1_{3 \times 3}]` is a selector matrix that selects the cartesian linear
    velocities from :math:`J_c(q)`, which is the Jacobian of the contact point between the wheel and
    the ground (in the world frame), and :math:`\dot{q}` are the joint velocities being optimized.

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            weight (float, np.array[float[3,3]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(PureRollingTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # TODO: check that the model is a wheeled robot

        raise NotImplementedError("This class has not been implemented yet.")
