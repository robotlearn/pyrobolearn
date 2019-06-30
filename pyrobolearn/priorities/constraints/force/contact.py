#!/usr/bin/env python
r"""Provide the contact (force normal) constraint.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import Constraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Songyan Xin (insight)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ContactConstraint(Constraint):
    r"""Contact force constraint

    The contact force constraint is given by :math:`0 \leq f^i_n` where :math:`f^i_n` is the normal force with respect
    to the contact surface applied on the link in contact :math:`i` defined in the local frame.

    References:
        - [1] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
    """

    def __init__(self, model):
        super(ContactConstraint, self).__init__(model)
