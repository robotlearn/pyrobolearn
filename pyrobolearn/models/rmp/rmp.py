#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Define the Riemannian Motion Policies

"The Riemannian Motion Policy (RMP) is an acceleration field (dynamical system) coupled with a corresponding Riemannian metric defining directions of importance at each point, typically defined on a nonlinear task space." [1]

References:
    - [1] "Riemannian Motion Policies", Ratliff et al., 2018
    - [2] "RMPflow: A Computational Graph for Automatic Motion Policy Generation", Cheng et al., 2018
    - [3] "Learning Reactive Motion Policies in Multiple Task Spaces from Human Demonstrations", Rana et al., 2019

Implementations found online:
    - RMP in ROS: https://github.com/rortiz9/rmp-ros
    - Multi-Robot RMPflow: https://github.com/gtrll/multi-robot-rmpflow
"""

# TODO: read and implement the RMP

import torch

from pyrobolearn.models.model import Model

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RMP(object):  # Model
    r"""Riemannian Motion Policies

    References:
        - [1] "Riemannian Motion Policies", Ratliff et al., 2018
        - [2] "RMPflow: A Computational Graph for Automatic Motion Policy Generation", Cheng et al., 2018
        - [3] "Learning Reactive Motion Policies in Multiple Task Spaces from Human Demonstrations", Rana et al., 2019
    """

    def __init__(self):
        pass

