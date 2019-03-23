#!/usr/bin/env python
"""Define the basic (Function) Approximator class.

This file describes the `Approximator` class that wraps a learning model and connects it with its inputs, and outputs.
The inputs/outputs can be states, actions, arrays/tensors, etc.

Dependencies:
- `pyrobolearn.models`
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

from abc import ABCMeta, abstractmethod
import collections
import numpy as np
import torch

from pyrobolearn.states import State
from pyrobolearn.actions import Action

from pyrobolearn.models import Model
from pyrobolearn.models.linear import Linear
from pyrobolearn.models.nn import NN, MLP, NEATModel

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Approximator(object):
    r"""Function Approximator (abstract) class

    The function approximator is a wrapper around the inner learning base model, and connects a learning model to
    its state/action inputs and outputs. That is, this class described how to connect a learning model with
    states/actions, and thus allows the inner models to be independent from the notions of states and actions.

    This class and all the children inheriting from this one will be used internally by several classes such as
    policies, dynamic models, value estimators, and others. This enables to not duplicate and write the same code
    for these various different concepts which share similar features.
    For instance, policies are function approximators where the inputs are states, and the outputs are actions.
    Dynamic models are extended models where the inputs are states and actions, and the outputs are the next state.

    Often the learning model can be constructed automatically by knowing the input and output dimensions, along with
    few other optional parameters. This is useful if one does not wish to change the learning model but just to scale
    it to different input/output sizes.

    This class is used by the following classes:
    * Policy: approximator mapping states to actions
    * Value estimator: approximator mapping states (or states and actions) to a value, or mapping states to actions
                       (in the case of discrete actions)
    * Actor-Critic:
    * Dynamic: approximator mapping states and actions to the next states
    * Transformation mappings: approximator mapping states to states

    Example::

        states = JntPosState(robot) + JntVelState(robot)
        actions = JntPosAction(robot)
        model = ...
    """

    def __init__(self, inputs, outputs, model=None, preprocessors=None, postprocessors=None):
        r"""Initialize the outer model.

        Args:
            inputs (State, Action, array): inputs of the inner models (instance of State/Action)
            outputs (State, Action, array): outputs of the inner models (instance of Action/State)
            model (Model, None): inner model which will be wrapped if not an instance of Model
            preprocessors (None, Processor): the inputs are first given to the preprocessors then to the model.
            postprocessors (None, Processor): the predicted outputs by the model are given to the processors before
                                              being returned.
        """

        # Check inputs and outputs, and convert to the correct format
        self._model = None
        self.inputs = inputs
        self.outputs = outputs

        # preprocessors and postprocessors
        if preprocessors is None:
            preprocessors = []
        if not isinstance(preprocessors, collections.Iterable):
            preprocessors = [preprocessors]
        self.preprocessors = preprocessors

        if postprocessors is None:
            postprocessors = []
        if not isinstance(postprocessors, collections.Iterable):
            postprocessors = [postprocessors]
        self.postprocessors = postprocessors

        # Check the given model: check if correct input/output sizes wrt the previous arguments, and check
        # the model type and wrap it if necessary. That is, if the type is from the original module/library,
        # wrap it with the corresponding inner model
        self.model = model

    ##############
    # Properties #
    ##############

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        if inputs is not None:
            if isinstance(inputs, (int, float)):
                inputs = np.array([inputs])
            elif not isinstance(inputs, (State, Action, torch.Tensor, np.ndarray)):
                raise TypeError("Expecting the inputs to be a State, Action, torch.Tensor, or np.ndarray.")
            if self._model is not None:
                pass  # TODO: check that the dimensions agree with the model
        # set inputs
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        if outputs is not None:
            if isinstance(outputs, (int, float)):
                outputs = np.array([outputs])
            elif not isinstance(outputs, (State, Action, torch.Tensor, np.ndarray)):
                raise TypeError("Expecting the outputs to be a State, Action, torch.Tensor, or np.ndarray.")
            if self._model is not None:
                pass  # TODO: check that the dimensions agree with the model
        # set outputs
        self._outputs = outputs

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model is not None:
            # check model type
            # if not isinstance(model, Model):
            #     raise TypeError("Expecting the model to be an instance of Model, instead received: "
            #                     "{}".format(type(model)))
            # TODO

            # check model input/output shape
            if self._inputs is None:
                raise ValueError("Inputs have not been set.")
            if self._outputs is None:
                raise ValueError("Outputs have not been set.")
            shape = model.input_shape

            # TODO

        # set model
        self._model = model

    @property
    def num_parameters(self):
        """Return the total number of parameters"""
        return self.model.num_parameters

    ###########
    # Methods #
    ###########

    def is_parametric(self):
        """
        Return True if the model is parametric.

        Returns:
            bool: True if the model is parametric.
        """
        return self.model.is_parametric()

    def is_linear(self):
        """
        Return True if the model is linear (wrt the parameters). This can be for instance useful for some learning
        algorithms (some only works on linear models).

        Returns:
            bool: True if it is a linear model
        """
        return self.model.is_linear()

    def is_recurrent(self):
        """
        Return True if the model is recurrent. This can be for instance useful for some learning algorithms which
        change their behavior when they deal with recurrent learning models.

        Returns:
            bool: True if it is a recurrent model.
        """
        return self.model.is_recurrent()

    def is_deterministic(self):
        """
        Return True if the model is deterministic.

        Returns:
            bool: True if the model is deterministic.
        """
        return self.model.is_deterministic()

    def is_probabilistic(self):
        """
        Return True if the model is probabilistic/stochastic.

        Returns:
            bool: True if the model is probabilistic.
        """
        return self.model.is_probabilistic()

    # alias
    is_stochastic = is_probabilistic

    def is_discriminative(self):
        """
        Return True if the model is discriminative, that is, if the model estimates the conditional probability
        :math:`p(y|x)`.

        Returns:
            bool: True if the model is discriminative.
        """
        return self.model.is_discriminative()

    def is_generative(self):
        """
        Return True if the model is generative, that is, if the model estimates the joint distribution of the input
        and output :math:`p(x,y)`. A generative model allows to sample from it.

        Returns:
            bool: True if the model is generative.
        """
        return self.model.is_generative()

    def reset(self):
        """Reset the approximator."""
        for processor in self.preprocessors:
            processor.reset()
        for processor in self.postprocessors:
            processor.reset()
        self.model.reset()

    def predict(self, x, to_numpy=True):
        for processor in self.preprocessors:
            x = processor(x)
        x = self.model(x.data[0])
        for processor in self.postprocessors:
            x = processor(x)
        return x

    def parameters(self):
        """Return the approximator parameters."""
        return self.model.parameters()

    def get_params(self):
        """Return the list of parameters."""
        return list(self.parameters())

    def hyperparameters(self):
        """Return the approximator hyper-parameters."""
        return self.model.hyperparameters()

    def get_hyperparams(self):
        """Return the list of hyper-parameters."""
        return list(self.hyperparameters())

    def get_vectorized_parameters(self, to_numpy=True):
        return self.model.get_vectorized_parameters(to_numpy=to_numpy)

    def set_vectorized_parameters(self, vector):
        self.model.set_vectorized_parameters(vector=vector)

    def get_input_dims(self):
        """Return the input dimensions."""
        return self.model.input_dims

    def get_output_dims(self):
        """Return the output dimensions."""
        return self.model.output_dims

    def save(self, filename):
        """save the inner model."""
        self.model.save(filename)

    def load(self, filename):
        """load the inner model."""
        self.model.load(filename)

    #############
    # Operators #
    #############

    def __call__(self, x):
        return self.predict(x)

    def __str__(self):
        return self.model.__str__()


class RandomApproximator(Approximator):
    r"""Random Approximator
    """

    class Random(object):

        def __init__(self, num_outputs, seed=None):
            self.num_outputs = num_outputs
            if seed is not None:
                np.random.seed(seed)

    def __init__(self, outputs, preprocessors=None, postprocessors=None):
        # call parent class
        model = self.Random(num_outputs=self._size(outputs), seed=None)
        super(RandomApproximator, self).__init__(inputs=None, outputs=outputs, model=model,
                                                 preprocessors=preprocessors, postprocessors=postprocessors)

    def _size(self, x):
        size = 0
        if isinstance(x, (State, Action)):
            if x.is_discrete():
                size = x.space[0].n
            else:
                size = x.total_size()
        elif isinstance(x, np.ndarray):
            size = x.size
        elif isinstance(x, torch.Tensor):
            size = x.numel()
        elif isinstance(x, int):
            size = x
        return size

    # def predict(self, x):
    #     if isinstance(self.outputs, (State, Action)):
    #         # get the space of each output
    #         spaces = self.outputs.space
    #
    #         # sample from each space
    #         output_data = [space.sample() for space in spaces]
    #
    #         # set the data for each action
    #         self.outputs.data = output_data
    #
    #         return self.outputs

    def predict(self, x):
        # x = self.preprocessors(x)
        x = self.model.predict(x.data[0])
        if isinstance(self.outputs, (State, Action)) and self.outputs.is_discrete():
            x = np.argmax(x)
        # x = self.postprocessors(x)
        return x


class LinearApproximator(Approximator):
    r"""Linear Function Approximator
    """

    def __init__(self, inputs, outputs, preprocessors=None, postprocessors=None):
        # call parent class
        model = Linear(num_inputs=self._size(inputs), num_outputs=self._size(outputs), add_bias=True)
        super(LinearApproximator, self).__init__(inputs, outputs, model=model, preprocessors=preprocessors,
                                                 postprocessors=postprocessors)

    def _size(self, x):
        size = 0
        if isinstance(x, (State, Action)):
            if x.is_discrete():
                size = x.space[0].n
            else:
                size = x.total_size()
        elif isinstance(x, np.ndarray):
            size = x.size
        elif isinstance(x, torch.Tensor):
            size = x.numel()
        elif isinstance(x, int):
            size = x
        return size

    def predict(self, x, to_numpy=True, return_logits=False):
        x = x.data[0]
        for processor in self.preprocessors:
            x = processor(x)
        x = self.model.predict(x, to_numpy=to_numpy)
        if isinstance(self.outputs, (State, Action)) and self.outputs.is_discrete() and not return_logits:
            if to_numpy:
                x = np.array([np.argmax(x)])
            else:
                x = torch.argmax(x, dim=0, keepdim=True)
        # x = self.postprocessors(x)
        return x


class NNApproximator(Approximator):
    r"""Neural Network Function Approximator
    """

    def __init__(self, inputs, outputs, model, preprocessors=None, postprocessors=None):

        # call parent class
        super(NNApproximator, self).__init__(inputs, outputs, model, preprocessors=preprocessors,
                                             postprocessors=postprocessors)

        # convert/wrap the model
        if not isinstance(model, NN):
            model = NN(model, input_dims=inputs.shape, output_dims=outputs.shape)  # TODO
        self.model = model


class MLPApproximator(NNApproximator):
    r"""Multi-Layer Perceptron Function Approximator

    It creates a feed-forward and fully-connected neural network, where linear layers are followed by non-linear
    activation functions. The input and output dimensions are inferred from the inputs and outputs.
    """

    def __init__(self, inputs, outputs, hidden_units=(),
                 activation_fct='Linear', last_activation_fct=None, dropout_prob=None,
                 preprocessors=None, postprocessors=None):

        # check that the inputs and ouputs are 1D
        # if not self._check1D(inputs):
        #     raise ValueError("Length of input shape should be 1! Instead, got {}".format(inputs.shape))
        # print(outputs)
        # print(outputs.shape)
        # if not self._check1D(outputs):
        #     raise ValueError("Length of output shape should be 1! Instead, got {}".format(outputs.shape))

        input_size = self._size(inputs)
        output_size = self._size(outputs)

        # create model
        num_units = [input_size] + list(hidden_units) + [output_size]
        model = MLP(num_units=num_units, activation_fct=activation_fct, last_activation_fct=last_activation_fct,
                    dropout_prob=dropout_prob)

        # call superclass
        super(MLPApproximator, self).__init__(inputs, outputs, model, preprocessors=preprocessors,
                                              postprocessors=postprocessors)

    def _size(self, x):
        if isinstance(x, (State, Action)):
            size = x.total_size()
        elif isinstance(x, np.ndarray):
            size = x.size
        elif isinstance(x, torch.Tensor):
            size = x.numel()
        elif isinstance(x, int):
            size = x
        return size

    def _check1D(self, arg):
        """Check that the given argument is a 1D vector, or simple array"""
        # if isinstance(arg, np.ndarray):
        shapes = arg.shape
        # else:
        #     shape = arg.shape()
        for shape in shapes:
            if not (len(shape) == 1 and isinstance(shape[0], int)):
                return False
        return True

    def predict(self, x):
        # convert given input to torch tensor
        if isinstance(x, (State, Action)):
            x = np.concatenate((x. data))
            x = torch.from_numpy(x).float()

        # feed it to the model and get predicted output
        x = self.model(x)

        # check output
        if isinstance(self.outputs, (State, Action)):
            # data
            self.outputs.train_data = x
            # convert back output from torch tensor to np array
            x = x.detach().numpy()
            self.outputs.data = x

            return self.outputs

        return x


class NEATApproximator(Approximator):
    r"""NEAT Approximator

    See Also: `neat_model.py`, `neat_policy`, `neat_algo.py`
    """

    def __init__(self, inputs, outputs, num_hidden=0, activation_fct='relu', network_type='feedforward',
                 aggregation='sum', weights_limits=(-20, 20), bias_limits=(-20, 20),
                 preprocessors=None, postprocessors=None):

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
        return self.model.genome

    @genome.setter
    def genome(self, genome):
        self.model.genome = genome

    @property
    def network(self):
        return self.model.network

    @property
    def population(self):
        return self.model.population

    ###########
    # Methods #
    ###########

    def _size(self, x):
        size = 0
        if isinstance(x, (State, Action)):
            if x.is_discrete():
                size = x.space[0].n
            else:
                size = x.total_size()
        elif isinstance(x, np.ndarray):
            size = x.size
        elif isinstance(x, torch.Tensor):
            size = x.numel()
        elif isinstance(x, int):
            size = x
        return size

    def predict(self, x, to_numpy=True):
        # x = self.preprocessors(x)
        x = self.model.predict(x.merged_data[0])

        if isinstance(self.outputs, (State, Action)):
            if self.outputs.is_discrete():
                x = np.argmax(x)
            elif self.outputs.is_continuous():
                x = 2 * np.array(x) - 1
            else:
                raise NotImplementedError("The outputs are not discrete or continuous...")
            self.outputs.data = x
        # x = self.postprocessors(x)
        # return x
        return self.outputs

    def set_network(self, genome=None, config=None):
        self.model.set_network(genome, config)

    def update_config(self, config):
        self.model.update_config(config)


# Tests
if __name__ == '__main__':
    pass
