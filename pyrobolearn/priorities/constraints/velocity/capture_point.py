# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the capture point constraint.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

# TODO: finish to implement this

import numpy as np

from pyrobolearn.priorities.constraints.constraint import UnilateralConstraint, JointVelocityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CapturePointConstraint(UnilateralConstraint, JointVelocityConstraint):
    r"""Capture Point constraint.

    Definition: "For a biped in state :math:`x`, a Capture Point (CP) :math:`P`, is a point on the ground such that
    if the biped covers :math:`P` (makes its base of support include :math:`P`), either with its stance foot or by
    stepping to :math:`P` in a single step, and then maintains its Center of Pressure (CoP) to lie on :math:`P`, then
    there exists a safe feasible trajectory leading to a capture state (i.e. a state in which the kinetic energy of
    the biped is zero and can remain zero with suitable joint torque (note that the CoM must lie above the CoP in a
    capture state))." [1] "Intuitively, the CP is the point on the floor onto which the robot has to step to come
    to a complete rest" [2].

    References:
        - [1] "Capture Point: A Step toward Humanoid PushRecovery", Pratt et al., 2006
        - [2] "Bipedal walking control based on Capture Point dynamics", Englsberger et al., 2011
    """

    def __init__(self, model):
        super(CapturePointConstraint, self).__init__(model)
