#!/usr/bin/env python
"""Define the Convolutional Neural Network (CNN) learning model.

This file provides the CNN model; a parametric, generally non-linear, non-recurrent, discriminative,
and deterministic model. This model is convenient for data arrays/tensors that have cells that have a spatial
relationship between them. For instance, pictures are 2D or 3D arrays where each pixel is related with its neighbors.

We decided to use the `pytorch` framework because of its popularity in the research community field, flexibility,
similarity with numpy (but with automatic differentiation: autograd), GPU capabilities, and more Pythonic approach.
While we thought about using other frameworks (such as Keras, Tensorflow, and others) as well, it would have
unnecessarily complexify the whole framework, as these frameworks would not only influence the learning models,
but also the losses, optimizers, and other modules. While we could have written some interfaces that makes the bridge
between these various frameworks and ours, we came to the conclusion that this would take a considerable amount of
efforts and time that we do not have for the moment.

References:
    [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    [2] PyTorch: https://pytorch.org/
"""

import copy
import inspect
import numpy as np
import torch

from pyrobolearn.models.nn.dnn import NN

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CNN(NN):
    r"""Convolutional Neural Network

    Feed-forward CNN.
    """
    pass


class CNNTorch(NNTorch):
    r"""Convolutional Neural Network in PyTorch
    """
    pass
