#!/usr/bin/env python
"""Define the basic (Function) Approximator class.

This file describes the `Approximator` class that wraps a learning model and connects it with its inputs, and outputs.
The inputs/outputs can be states, actions, arrays/tensors, etc.

Dependencies:
- `pyrobolearn.models`
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

import copy
import collections
import numpy as np
import torch

from pyrobolearn.states import State
from pyrobolearn.actions import Action
from pyrobolearn.models import Model
from pyrobolearn.processors import Processor


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
    its state / action inputs and outputs. That is, this class described how to connect a learning model with
    states / actions, and thus allows the inner models to be independent from the notions of states and actions.

    This class and all the children inheriting from this one will be used internally by several classes such as
    policies, dynamic models, value estimators, and others. This enables to not duplicate and write the same code
    for these various different concepts which share similar features. For instance, policies are function
    approximators where the inputs are states, and the outputs are actions while dynamic transition models are
    approximators where the inputs are states and actions, and the outputs are the next state.

    Often the learning model can be constructed automatically by knowing the input and output dimensions, along with
    few other optional parameters. This is useful if one does not wish to change the learning model but just to scale
    it to different input/output sizes.

    This class is used by the following classes:
    * Policy: approximator mapping states to actions
    * Value estimator: approximator mapping states (or states and actions) to a value, or mapping states to actions
                       (in the case of discrete actions)
    * Actor-Critic: combination of policy and value function approximators
    * Dynamic: approximator mapping states and actions to the next states
    * Transformation mappings: approximator mapping states to states
    """

    def __init__(self, inputs, outputs, model=None, preprocessors=None, postprocessors=None):
        r"""Initialize the outer model.

        Args:
            inputs ((list of) State, Action, np.array, torch.Tensor): inputs of the inner models (instance of
                State / Action)
            outputs (State, Action, np.array, torch.Tensor): outputs of the inner models (instance of Action/State)
            model (Model, None): inner model which will be wrapped if not an instance of Model
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
                to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
                processors before being returned.
        """

        # Check inputs and outputs, and convert to the correct format
        self._model = None
        self.inputs = inputs
        self.outputs = outputs

        # set pre- and post- processors
        self.preprocessors = preprocessors
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
        """Return the approximator's inputs."""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        """Set the approximator's inputs."""
        if inputs is not None:
            if isinstance(inputs, (int, float)):
                inputs = np.array([inputs])
            elif isinstance(inputs, list):
                for x in inputs:
                    if not isinstance(x, (State, Action, torch.Tensor, np.ndarray)):
                        raise TypeError("Expecting the given input to be an instance of `State`, `Action`, "
                                        "`torch.Tensor`, `np.ndarray`, instead got: {}".format(type(x)))
            elif not isinstance(inputs, (State, Action, torch.Tensor, np.ndarray)):
                raise TypeError("Expecting the inputs to be a State, Action, torch.Tensor, np.ndarray, or a list "
                                "of them.")
            if self._model is not None:
                pass  # TODO: check that the dimensions agree with the model
        # set inputs
        self._inputs = inputs

    @property
    def outputs(self):
        """Return the approximator's outputs."""
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        """Set the approximator's outputs."""
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
        """Return the inner learning model."""
        return self._model

    @model.setter
    def model(self, model):
        """Set the inner learning model."""
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
    def preprocessors(self):
        """Return the list of pre-processors."""
        return self._preprocessors

    @preprocessors.setter
    def preprocessors(self, processors):
        """Set the list of pre-processors."""
        if processors is None:
            processors = []
        elif callable(processors):
            processors = [processors]
        elif isinstance(processors, collections.Iterable):
            for idx, processor in enumerate(processors):
                if not callable(processor):
                    raise ValueError("The {} processor {} is not callable.".format(idx, processor))
        else:
            raise TypeError("Expecting the processors to be None, a callable class / function such as `Processor`, "
                            "or a list of them. Instead got: {}".format(type(processors)))
        self._preprocessors = processors

    @property
    def postprocessors(self):
        """Return the list of post-processors."""
        return self._postprocessors

    @postprocessors.setter
    def postprocessors(self, processors):
        """Set the list of post-processors."""
        if processors is None:
            processors = []
        elif callable(processors):
            processors = [processors]
        elif isinstance(processors, collections.Iterable):
            for idx, processor in enumerate(processors):
                if not callable(processor):
                    raise ValueError("The {} processor {} is not callable.".format(idx, processor))
        else:
            raise TypeError("Expecting the processors to be None, a callable class / function such as `Processor`, "
                            "or a list of them. Instead got: {}".format(type(processors)))
        self._postprocessors = processors

    @property
    def input_size(self):
        """Return the approximator input size."""
        return self.model.input_size

    @property
    def output_size(self):
        """Return the approximator output size."""
        return self.model.output_size

    @property
    def input_shape(self):
        """Return the approximator input shape."""
        return self.model.input_shape

    @property
    def output_shape(self):
        """Return the approximator output shape."""
        return self.model.output_shape

    @property
    def input_dim(self):
        """Return the input dimension."""
        return self.model.input_dim

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self.model.output_dim

    @property
    def num_parameters(self):
        """Return the total number of parameters of the inner learning model."""
        return self.model.num_parameters

    @property
    def num_hyperparameters(self):
        """Return the total number of hyper-parameters of the inner learning model."""
        return self.model.num_hyperparameters

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

    @staticmethod
    def _size(x):
        """Return the total size of a `State`, `Action`, numpy.array, or torch.Tensor."""
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

    def train(self):
        """Set the inner model into training mode."""
        self.model.train()

    def eval(self):
        """Set the inner model into testing mode."""
        self.model.eval()

    def parameters(self):
        """Return an iterator over the approximator parameters."""
        return self.model.parameters()

    def named_parameters(self):
        """Return an iterator over the approximator parameters, yielding both the name and the parameter itself."""
        return self.model.named_parameters()

    def list_parameters(self):
        """Return the list of parameters."""
        return list(self.parameters())

    def hyperparameters(self):
        """Return an iterator over the approximator hyper-parameters."""
        return self.model.hyperparameters()

    def named_hyperparameters(self):
        """Return an iterator over the approximator hyper-parameters, yielding both the name and the hyper-parameter
        itself."""
        return self.model.named_hyperparameters()

    def list_hyperparameters(self):
        """Return the list of hyper-parameters."""
        return list(self.hyperparameters())

    def get_vectorized_parameters(self, to_numpy=True):
        """Return a vectorized form of the parameters"""
        return self.model.get_vectorized_parameters(to_numpy=to_numpy)

    def set_vectorized_parameters(self, vector):
        """Set the vector parameters."""
        self.model.set_vectorized_parameters(vector=vector)

    def reset(self, reset_processors=False):
        """Reset the approximators."""
        if reset_processors:
            for processor in self.preprocessors:
                if isinstance(processor, Processor):
                    processor.reset()
            for processor in self.postprocessors:
                if isinstance(processor, Processor):
                    processor.reset()
        self.model.reset()

    @staticmethod
    def __convert_to_numpy(x, to_numpy=True):
        """Convert the given argument to a numpy array if specified."""
        if to_numpy and isinstance(x, torch.Tensor):
            if x.requires_grad:
                return x.detach().numpy()
            return x.numpy()
        return x

    @staticmethod
    def merge_inputs(x=None, to_numpy=True):
        """
        Merge the inputs of the approximator.

        Args:
            x (None, (list of) State / Action / np.array / torch.Tensor): input data. If None, it will get the
                data from the inputs that were given at the initialization.
            to_numpy (bool): If True, it will convert to numpy arrays.

        Returns:
            list of np.array / torch.Tensor: input data
        """
        # if no input is given, take the provided inputs at the beginning
        if x is None:
            pass

    def predict(self, x=None, to_numpy=True, return_logits=False, set_output_data=True):
        """Predict the output given the input.

        Args:
            x (None, State, Action, (list of) np.array, (list of) torch.Tensor): input data. If None, it will get the
                data from the inputs that were given at the initialization.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
            return_logits (bool): If True, in the case of discrete outputs, it will return the logits.
            set_output_data (bool): If True, it will set the predicted output data to the outputs given at the
                initialization.

        Returns:
            (list of) np.array, list of (torch.Tensor): predicted output data.
        """
        # if no input is given, take the provided inputs at the beginning
        if x is None:
            x = self.inputs

        # if the input is an instance of State or Action, get the inner merged data.
        if isinstance(x, (State, Action)):
            x = x.merged_data
            if len(x) == 1:
                x = x[0]

        # go through each preprocessor
        for processor in self.preprocessors:
            x = processor(x)

        # predict output using the learning model
        x = self.model.predict(x, to_numpy=False)

        # go through each postprocessor
        for processor in self.postprocessors:
            x = processor(x)

        # set the output data and convert it if specified
        if isinstance(self.outputs, (State, Action)) and (to_numpy or not return_logits or set_output_data):
            # if output data `x` is not a list, make it a list as we will iterate through it
            if not isinstance(x, list):
                x = [x]

            # go through each output and output data
            for idx, (output, data) in enumerate(zip(self.outputs, x)):
                if isinstance(data, np.ndarray):
                    if output.is_discrete():
                        discrete_data = np.array([np.argmax(data)])
                        if set_output_data:
                            output.data = discrete_data
                        if not return_logits:
                            x[idx] = discrete_data
                    elif set_output_data:
                        output.data = data
                elif isinstance(data, torch.Tensor):
                    if output.is_discrete():
                        discrete_data = torch.argmax(data, dim=0, keepdim=True)
                        if set_output_data:
                            output.torch_data = discrete_data
                        if return_logits:
                            x[idx] = self.__convert_to_numpy(data, to_numpy=to_numpy)
                        else:
                            x[idx] = self.__convert_to_numpy(discrete_data, to_numpy=to_numpy)
                else:
                    raise TypeError("Expecting `data` output to be a numpy array, torch.Tensor, or a list of them, "
                                    "instead got: {}".format(type(data)))

        # if output is a list and has one element, return just that element
        if isinstance(x, list) and len(x) == 1:
            x = x[0]

        # return the output data
        return x

    def save(self, filename):
        """Save the inner model on the disk.

        Args:
            filename (str): path to the file to save the model.
        """
        self.model.save(filename)

    def load(self, filename):
        """Load the inner model from the disk.

        Args:
            filename (str): path to the file which contains the model.
        """
        # TODO: check if model is None
        self.model = self.model.load(filename)
        return self.model

    #############
    # Operators #
    #############

    def __call__(self, x):
        """Predict the output using the inner learning model given the input."""
        return self.predict(x)

    # def __repr__(self):
    #     """Return a representation of the model."""
    #     return self.model.__str__()

    def __str__(self):
        """Return a string describing the model."""
        return self.model.__str__()

    def __copy__(self):
        """Return a shallow copy of the approximator. This can be overridden in the child class."""
        return self.__class__(inputs=self.inputs, outputs=self.outputs, model=self.model,
                              preprocessors=self.preprocessors, postprocessors=self.postprocessors)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the approximator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        def get_inputs_outputs(items):
            if isinstance(items, list):
                elements = []
                for item in items:
                    if isinstance(item, (Action, State)):
                        elements.append(copy.deepcopy(item, memo))
                    else:
                        elements.append(copy.deepcopy(item))
            elif isinstance(items, (Action, State)):
                elements = copy.deepcopy(items, memo)
            else:
                elements = copy.deepcopy(items)
            return elements

        inputs = get_inputs_outputs(self.inputs)
        outputs = get_inputs_outputs(self.outputs)
        model = copy.deepcopy(self.model, memo) if self.model is not None else None
        preprocessors = [copy.deepcopy(preprocessor, memo) for preprocessor in self.preprocessors]
        postprocessors = [copy.deepcopy(postprocessor, memo) for postprocessor in self.postprocessors]
        approximator = self.__class__(inputs=inputs, outputs=outputs, model=model,
                                      preprocessors=preprocessors, postprocessors=postprocessors)

        # update the memodict (note that `copy.deepcopy` will automatically check this dictionary and return the
        # reference if already present)
        memo[self] = approximator

        return approximator


# Tests
if __name__ == '__main__':
    pass
