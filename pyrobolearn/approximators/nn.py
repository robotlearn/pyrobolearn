#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define Neural Network approximators.

Dependencies:
- `pyrobolearn.models.nn`
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

import collections
import numpy as np
import torch

from pyrobolearn.states import State
from pyrobolearn.actions import Action

from pyrobolearn.approximators.approximator import Approximator
from pyrobolearn.models.nn import NN, MLP, NEATModel

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NNApproximator(Approximator):
    r"""Neural Network Function Approximator
    """

    def __init__(self, inputs, outputs, model, preprocessors=None, postprocessors=None):
        """
        Initialize the Neural Network approximator.

        Args:
            inputs (State, Action, np.array, torch.Tensor): inputs of the inner models (instance of State/Action)
            outputs (State, Action, np.array, torch.Tensor): outputs of the inner models (instance of Action/State)
            model (Model, torch.nn.Module): Learning model
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
                to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
                processors before being returned.
        """

        # call parent class
        super(NNApproximator, self).__init__(inputs, outputs, model, preprocessors=preprocessors,
                                             postprocessors=postprocessors)

        # convert/wrap the model
        if not isinstance(model, NN):
            model = NN(model, input_shape=inputs.shape, output_shape=outputs.shape)  # TODO
        self.model = model


class MLPApproximator(NNApproximator):
    r"""Multi-Layer Perceptron Function Approximator

    It creates a feed-forward and fully-connected neural network, where linear layers are followed by non-linear
    activation functions. The input and output dimensions are inferred from the inputs and outputs.
    """

    def __init__(self, inputs, outputs, hidden_units=(),
                 activation='Linear', last_activation=None, dropout=None,
                 preprocessors=None, postprocessors=None):
        """
        Initialize the Multi-Layer Perceptron approximator.

        Args:
            inputs (State, Action, np.array, torch.Tensor): inputs of the inner models (instance of State/Action)
            outputs (State, Action, np.array, torch.Tensor): outputs of the inner models (instance of Action/State)
            hidden_units (tuple, list of int): number of hidden units in each layer
            activation (str): activation function to apply on each layer
            last_activation (str, None): activation function to apply on the last layer
            dropout (None, float): dropout probability
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
                to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
                processors before being returned.
        """

        # check that the inputs and ouputs are 1D
        # if not self._check_1d(inputs):
        #     raise ValueError("Length of input shape should be 1! Instead, got {}".format(inputs.shape))
        # print(outputs)
        # print(outputs.shape)
        # if not self._check_1d(outputs):
        #     raise ValueError("Length of output shape should be 1! Instead, got {}".format(outputs.shape))

        input_size = self._size(inputs)
        output_size = self._size(outputs)

        # create model
        units = [input_size] + list(hidden_units) + [output_size]
        model = MLP(units=units, activation=activation, last_activation=last_activation, dropout=dropout)

        # call superclass
        super(MLPApproximator, self).__init__(inputs, outputs, model, preprocessors=preprocessors,
                                              postprocessors=postprocessors)

    @staticmethod
    def _check_1d(arg):
        """Check that the given argument is a 1D vector, or simple array"""
        # if isinstance(arg, np.ndarray):
        shapes = arg.shape
        # else:
        #     shape = arg.shape()
        for shape in shapes:
            if not (len(shape) == 1 and isinstance(shape[0], int)):
                return False
        return True

    # def predict(self, x):
    #     # convert given input to torch tensor
    #     if isinstance(x, (State, Action)):
    #         x = x.merged_data[0]
    #         x = torch.from_numpy(x).float()
    #
    #     # feed it to the model and get predicted output
    #     x = self.model(x)
    #
    #     # check output
    #     if isinstance(self.outputs, (State, Action)):
    #         # data
    #         self.outputs.train_data = x
    #         # convert back output from torch tensor to np array
    #         x = x.detach().numpy()
    #         self.outputs.data = x
    #         return self.outputs
    #     return x


class NEATApproximator(Approximator):
    r"""NEAT Approximator

    See Also: `neat_model.py`, `neat_policy`, `neat_algo.py`
    """

    def __init__(self, inputs, outputs, num_hidden=0, activation_fct='relu', network_type='feedforward',
                 aggregation='sum', weights_limits=(-20, 20), bias_limits=(-20, 20),
                 preprocessors=None, postprocessors=None):
        """
        Initialize the NEAT Approximator. This uses as the inner learning model a neural network that can evolve its
        weights as well as its topology.

        Args:
            inputs (int, np.array, torch.Tensor, State, Action): inputs.
            outputs (int, np.array, torch.Tensor, State, Action): outputs.
            num_hidden (int): number of hidden units.
            activation_fct (str): activation function to use.
            network_type (str): type of neural network. Select between 'feedforward' and 'recurrent'.
            aggregation (str): how to aggregate the input signals of a node. Select between 'sum', 'product', 'max',
                'min', 'maxabs', 'median', and 'mean'.
            weights_limits (tuple): weight limits / bounds. The tuple contains the lower and upper bounds.
            bias_limits (tuple): bias limits / bounds. The tuple contains the lower and upper bounds.
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
                to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
                processors before being returned.
        """

        # call parent class
        model = NEATModel(num_inputs=self._size(inputs), num_outputs=self._size(outputs), num_hidden=num_hidden,
                          activation_fct=activation_fct, network_type=network_type, aggregation=aggregation,
                          weights_limits=weights_limits, bias_limits=bias_limits)
        super(NEATApproximator, self).__init__(inputs, outputs, model=model, preprocessors=preprocessors,
                                               postprocessors=postprocessors)

    ##############
    # Properties #
    ##############

    @property
    def config(self):
        """Return the config object"""
        return self.model.config

    @config.setter
    def config(self, config):
        """Set the config file (str) or object."""
        self.model.config = config

    @property
    def genome(self):
        """Return the NEAT model's genome."""
        return self.model.genome

    @genome.setter
    def genome(self, genome):
        """Set the genome."""
        self.model.genome = genome

    @property
    def network(self):
        """Return the NEAT model's network."""
        return self.model.network

    @property
    def population(self):
        """Return the population used in NEAT."""
        return self.model.population

    ###########
    # Methods #
    ###########

    # def predict(self, x, to_numpy=True, return_logits=True, set_output_data=False):
    #     # x = self.preprocessors(x)
    #     x = self.model.predict(x)
    #
    #     if isinstance(self.outputs, (State, Action)):
    #         if self.outputs.is_discrete():
    #             print(x)
    #             x = np.argmax(x)
    #             print(x)
    #         elif self.outputs.is_continuous():
    #             x = 2 * np.array(x) - 1
    #         else:
    #             raise NotImplementedError("The outputs are not discrete or continuous...")
    #         self.outputs.data = x
    #     # x = self.postprocessors(x)
    #     # return x
    #     return self.outputs

    def set_network(self, genome=None, config=None):
        """Set the genome network."""
        self.model.set_network(genome, config)

    def update_config(self, config):
        """Update the configuration file."""
        self.model.update_config(config)
