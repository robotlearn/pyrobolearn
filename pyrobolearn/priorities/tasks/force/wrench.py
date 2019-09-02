#!/usr/bin/env python
r"""Provide the wrench task.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import Task


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WrenchTask(Task):
    r"""Wrench Task

    The wrench task

    """

    def __init__(self, model, constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface
            constraints (list of Constraint): list of constraints associated with the task.
        """
        super(WrenchTask, self).__init__(model=model, constraints=constraints)
