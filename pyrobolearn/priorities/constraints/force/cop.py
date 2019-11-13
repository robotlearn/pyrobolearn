#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the center of pressure constraint.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import Constraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Arturo Laurenzi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CoPConstraint(Constraint):
    r"""Center of Pressure (CoP) constraint.

    "The CoP is the point on the ground where the resultant of the ground-reaction force acts". [1]

    This is defined mathematically as:

    .. math::

        x_{CoP} = \frac{\sum_i x_i f^i_n}{\sum_i f^i_n}
        y_{CoP} = \frac{\sum_i y_i f^i_n}{\sum_i f^i_n}
        z_{CoP} = \frac{\sum_i z_i f^i_n}{\sum_i f^i_n}

    where :math:`[x_i, y_i, z_i]` are the coordinates of the contact point :math:`i` on which the normal force
    :math:`f^i_n` acts.

    Notes:
        - the ZMP and CoP are equivalent for horizontal ground surfaces. For irregular ground surfaces they are
        distinct. [2]

    References:
        - [1] "Postural Stability of Biped Robots and Foot-Rotation Index (FRI) Point", Goswami, 1999
        - [2] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
        Implications", Popovic et al., 2005
    """

    def __init__(self, model):
        super(CoPConstraint, self).__init__(model)
