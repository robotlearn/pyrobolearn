#!/usr/bin/env python
"""Define the Recurrent Convolutional Neural Network (RCNN) learning model.

This file provides the RCNN model; a parametric, generally non-linear, recurrent, discriminative,
and deterministic model. This model is convenient for sequential data arrays/tensors where at each instant, the cells
in the data arrays/tensors have a spatial relationship between them. For instance, this model can be used with videos
where there is a temporal and spatial relationship between the pixels.

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

from dnn import NN

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RCNN(NN):
    r"""Recurrent CNN
    """
    pass

class RCNN(NNTorch):
    r"""Recurrent CNN in PyTorch
    """
    pass
