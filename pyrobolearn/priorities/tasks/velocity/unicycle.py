#!/usr/bin/env python
r"""Provide the unicycle task.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

# TODO: finish to implement this class

import numpy as np

from pyrobolearn.priorities.tasks import Task


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Juan Alejandro Castano (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class UnicycleTask(Task):
    r"""Unicycle Task

    The unicycle task defines a rotation around fix axes to allow the robot wheels to spin. It basically creates a
    new Cartesian tasks at the wheels.

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface
            weight (float, np.array[N,N]): weight scalar or matrix associated to the task.
            constraints (list of Constraint): list of constraints associated with the task.
        """
        super(UnicycleTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # TODO: check that the model is a wheeled robot

        raise NotImplementedError("This class has not been implemented yet.")
