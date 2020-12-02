#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the knowledge distillation task.

This type of tasks takes one or multiple models that have been trained on one or several tasks, and use them to train
a single model to compress the acquired knowledge. The target model can also be smaller than the source model(s)
reducing the space and possibly the time complexity.

References:
    [1] "Distilling the Knowledge in a Neural Network", Hinton et al., 2015
"""

import collections.abc
import torch

from pyrobolearn.models.model import Model
from pyrobolearn.approximators.approximator import Approximator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DistillationTask(object):
    r"""Knowledge Distillation Task

    This type of tasks takes one or multiple models that have been trained on one or several tasks, and use them to
    train a single model to compress the acquired knowledge. The target model can also be smaller than the source
    model(s) reducing the space and possibly the time complexity.

    References:
        [1] "Distilling the Knowledge in a Neural Network", Hinton et al., 2015
    """

    def __init__(self, source_models, target_model, datasets=None):
        """
        Initialize the Distillation task.

        Args:
            source_models ((list of) Approximator / Model / torch.nn.Module): source approximators / learning models.
            target_model (Approximator / Model / torch.nn.Module): target approximator / learning model.
            datasets (list of Dataset, Dataset): dataset to which train the target approximator / learning model on.
        """
        self.source_models = source_models
        self.target_model = target_model
        self.datasets = datasets

    ##############
    # Properties #
    ##############

    @property
    def source_models(self):
        """Return the source models which possess the knowledge."""
        return self._source_models

    @source_models.setter
    def source_models(self, models):
        """Set the source models."""
        if not isinstance(models, (list, tuple, set)):
            models = [models]
        for model in models:
            if not isinstance(model, (Model, Approximator, torch.nn.Module)):
                raise TypeError("Expecting the given source model to be an instance of `Model`, `Approximator`, or "
                                "`torch.nn.Module`, instead got: {}".format(type(model)))
        self._source_models = models

    @property
    def target_model(self):
        """Return the target model which will contained the distilled knowledge once trained."""
        return self._target_model

    @target_model.setter
    def target_model(self, model):
        """Set the target model."""
        if not isinstance(model, (Model, Approximator, torch.nn.Module)):
            raise TypeError("Expecting the given target model to be an instance of `Model`, `Approximator`, or "
                            "`torch.nn.Module`, instead got: {}".format(type(model)))
        self._target_model = model

    @property
    def datasets(self):
        """Return the datasets."""
        return self._datasets

    @datasets.setter
    def datasets(self, datasets):
        if not isinstance(datasets, (list, tuple, set)):
            datasets = [datasets]
        if len(datasets) != len(self.source_models):
            raise ValueError("The number of datasets (={}) does not match the number of source models (={})"
                             ".".format(len(datasets), len(self.source_models)))
        for dataset in datasets:
            if not isinstance(dataset, torch.utils.data.Dataset):
                raise TypeError("Expecting the dataset to be an instance of `Dataset`, or `torch.utils.data.Dataset`,"
                                " instead got: {}".format(type(dataset)))
        self._datasets = datasets

    ###########
    # Methods #
    ###########

    def train(self, datasets=None, method=None):
        """
        Train the target model on the provided dataset(s) and the predicted output from the source models.

        Args:
            datasets (None): If None, it will use the original datasets given at the initialization.
            method (None): specify which method to use to distill the knowledge.
        """
        pass
