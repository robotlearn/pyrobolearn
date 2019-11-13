#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the gaze task.


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
__credits__ = ["Enrico Mingo Hoffman (C++)", "Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GazeTask(JointVelocityTask):
    r"""Gaze Task

    The Gaze class implement a Cartesian Task in which the gaze of the robot is controlled. This is achieved by
    controlling the orientation of the distal link equipped with a camera (this can for instance be the head of a
    robot) with respect to a base frame (which can be the neck or waist for instance). For this purpose, from the
    Cartesian sub-task only the pitch and yaw velocities are considered.

    The implementation is based on [1] which itself is inspired from [2].

    References:
        - [1] OpenSoT framework
        - [2] "Adaptive Predictive Gaze Control of a Redundant Humanoid Robot Head", Milighetti et al., 2011
    """

    def __init__(self, model, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            weight (float, np.array[float[2,2]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(GazeTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # self.cartesian_task = CartesianTask(self.model, distal_link=distal_link, weight=weight)

        raise NotImplementedError("This class has not been implemented yet.")
