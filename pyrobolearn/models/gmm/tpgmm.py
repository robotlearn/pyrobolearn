#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Task-Parametrized Gaussian Mixture Model (TP-GMM).

This file provides the Task-parametrized GMM.
"""

# TODO: implement TP-GMM which should be pretty straightforward. Just need to have multiple GMMs, make the necessary transforms,
# and use Gaussian product when predicting.

import numpy as np
from pyrobolearn.models.gmm.gmm import GMM

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TPGMM(GMM):
    r"""Task-Parametrized Gaussian Mixtured Model

    References:
        - [1] "A Tutorial on Task-Parameterized Movement Learning and Retrieval", Calinon, 2015
    """

    def __init__(self, num_frames, num_components):
        super(TPGMM, self).__init__(num_components)
        self.num_systems = num_frames
