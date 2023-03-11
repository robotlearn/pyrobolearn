#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide Capsule networks.

References:
    [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    [2] PyTorch: https://pytorch.org/
    [3] "Dynamic Routing Between Capsules", Sabour et al., 2017 (https://arxiv.org/abs/1710.09829)
    [4] Medium blog post: "Introducing awesome-capsule-networks"
        https://medium.com/@sekwiatkowski/introducing-awesome-capsule-networks-897f1b81d1e3

Interesting implementations:
- https://github.com/gram-ai/capsule-networks
- https://github.com/danielhavir/capsule-network
- https://github.com/higgsfield/Capsule-Network-Tutorial
"""

import torch

from pyrobolearn.models.nn.dnn import NN

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: implement
class CapsuleNN(NN):
    r"""Capsule Neural Networks"""
    pass
