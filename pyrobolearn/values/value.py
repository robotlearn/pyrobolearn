#!/usr/bin/env python
"""Provides the various abstract value function approximators.

It can use learning models as most of these models learn a function (that is how to map inputs to outputs).
For instance, a value network is a function represented by a neural network that maps a state to a value (real number).

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
- `pyrobolearn.approximators` (and thus `pyrobolearn.models`)
"""

import copy
from abc import ABCMeta, abstractmethod
import numpy as np
import torch

from pyrobolearn.states import State
from pyrobolearn.actions import Action
from pyrobolearn.approximators import Approximator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ValueApproximator(object):
    r"""Value Function Approximator

    In reinforcement learning, this is known as value-based reinforcement learning. The other end being policy
    search. Several methods fall in the spectrum between these 2 approaches. For instance, actor-critic methods
    optimize a policy (aka the actor) using the value function (aka critic).

    As for the Policy (see `policy.py`), the value function approximator groups a model (which can be learned as a
    neural network, or built as a table) with the states and actions.

    There are mainly two types of value function:
    1. state value function denoted by :math:`V(s)` which represents the expected reward accumulated when
       starting at state :math:`s`.
    2. (state-)action value function denoted by :math:`Q(s,a)`, which represents the expected reward when
       starting at state :math:`s`, and performing action :math:`a`.

    In both cases, they use the state. This class computes :math:`V(s)`.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state):
        """
        Initialize the value function approximator.

        Args:
            state (State, np.array, torch.Tensor): state input
        """
        self.state = state
        self.value = None

    ##############
    # Properties #
    ##############

    @property
    def state(self):
        """Return the state instance."""
        return self._state

    @state.setter
    def state(self, state):
        """Set the state input."""
        if isinstance(state, (int, float)):
            state = np.array([state])
        elif not isinstance(state, (State, torch.Tensor, np.ndarray)):
            raise TypeError("Expecting the state to be a State, torch.Tensor, or np.ndarray.")
        self._state = state

    ###########
    # Methods #
    ###########

    def reset(self):
        """Reset the value approximator."""
        pass

    def evaluate(self, *args, **kwargs):
        """Predict the value."""
        pass

    #############
    # Operators #
    #############

    # def __repr__(self):
    #     """Return a representation string about the reward function."""
    #     return self.__class__.__name__

    def __str__(self):
        """Return a string describing the reward function."""
        return self.__repr__()

    def __call__(self, *args, **kwargs):
        """Predict the value."""
        return self.evaluate(*args, **kwargs)

    def __copy__(self):
        """Return a shallow copy of the value approximator. This can be overridden in the child class."""
        return self.__class__(state=self.state)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the value approximator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        state = copy.deepcopy(self.state, memo) if isinstance(self.state, State) else copy.deepcopy(self.state)
        value = self.__class__(state=state)
        memo[self] = value
        return value


class QValueApproximator(ValueApproximator):
    r"""Q-Value Approximator

    Compute :math:`Q(s,a)`.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, action):
        """
        Initialize the value function approximator.

        Args:
            state (State, np.array, torch.Tensor): state input
            action (Action, np.array, torch.Tensor): action input (if discrete or continuous) or output (only if
                discrete).
        """
        # super(QValueApproximator, self).__init__(state)
        ValueApproximator.__init__(self, state)
        self.action = action

    ##############
    # Properties #
    ##############

    @property
    def action(self):
        """Return the action instance."""
        return self._action

    @action.setter
    def action(self, action):
        """Set the action input."""
        if isinstance(action, (int, float)):
            action = np.array([action])
        elif not isinstance(action, (Action, torch.Tensor, np.ndarray)):
            raise TypeError("Expecting the action to be an Action, torch.Tensor, or np.ndarray.")
        self._action = action

    #############
    # Operators #
    #############

    def __copy__(self):
        """Return a shallow copy of the value approximator. This can be overridden in the child class."""
        return self.__class__(state=self.state, action=self.action)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the value approximator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        state = copy.deepcopy(self.state, memo) if isinstance(self.state, State) else copy.deepcopy(self.state)
        action = copy.deepcopy(self.action, memo) if isinstance(self.action, Action) else copy.deepcopy(self.action)
        value = self.__class__(state=state, action=action)
        memo[self] = value
        return value


class ParametrizedValue(ValueApproximator):
    r"""Parametrized (Learnable) Value Function Approximator

    This value function approximator has parameters that can be optimized. By default it predicts :math:`V_{\phi}(s)`.
    """

    def __init__(self, state, model):
        """
        Initialize the parametrized value function approximator.

        Args:
            state (State): input state.
            model (Approximator): value function approximator.
        """
        # ValueApproximator.__init__(self, state)
        super(ParametrizedValue, self).__init__(state)
        self.model = model

    ##############
    # Properties #
    ##############

    @property
    def model(self):
        """Return the model instance."""
        return self._model

    @model.setter
    def model(self, model):
        """Set the model / approximator instance."""
        if not isinstance(model, Approximator):
            raise TypeError("Expecting the model to be an instance of `Approximator`, instead got: "
                            "{}".format(type(model)))
        self._model = model

    @property
    def input_size(self):
        """Return the policy input size."""
        return self.model.input_size

    @property
    def output_size(self):
        """Return the policy output size."""
        return self.model.output_size

    @property
    def input_shape(self):
        """Return the policy input shape."""
        return self.model.input_shape

    @property
    def output_shape(self):
        """Return the policy output shape."""
        return self.model.output_shape

    @property
    def input_dim(self):
        """Return the input dimension of the policy; i.e. len(input_shape)."""
        return self.model.input_dim

    @property
    def output_dim(self):
        """Return the output dimension of the policy; i.e. len(output_shape)."""
        return self.model.output_dim

    @property
    def num_parameters(self):
        """Return the total number of parameters"""
        return self.model.num_parameters

    ###########
    # Methods #
    ###########

    def parameters(self):
        """
        Return an iterator over the learning model parameters.
        """
        return self.model.parameters()

    def named_parameters(self):
        """
        Return an iterator over the learning model parameters; yielding both the name and the parameter itself.
        """
        return self.model.named_parameters()

    def list_parameters(self):
        """
        Return the learning model parameters.
        """
        return self.model.list_parameters()

    def hyperparameters(self):
        """
        Return an iterator over the learning model hyper-parameters.
        """
        return self.model.hyperparameters()

    def named_hyperparameters(self):
        """
        Return an iterator over the learning model hyper-parameters; yielding both the name and the hyper-parameter
        itself.
        """
        return self.model.named_hyperparameters()

    def list_hyperparameters(self):
        """
        Return the learning model hyper-parameters
        """
        return self.model.list_hyperparameters()

    def get_vectorized_parameters(self, to_numpy=True):
        """
        Get the parameters in a vectorized form.

        Args:
            to_numpy (bool): if True, it will convert the 1D parameter vector into a numpy array.
        """
        return self.model.get_vectorized_parameters(to_numpy=to_numpy)

    def set_vectorized_parameters(self, vector):
        """
        Set the vectorized parameters.

        Args:
            np.array, torch.Tensor: 1D parameter vector.
        """
        self.model.set_vectorized_parameters(vector=vector)

    def evaluate(self, state=None, to_numpy=False):
        """Compute the output of the value function.

        Args:
            state (None, State, (list of) np.array, (list of) torch.Tensor): state input data. If None, it will get
                the data from the inputs that were given at the initialization.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
        """
        # if no input is given, take the provided inputs at the beginning
        if state is None:
            state = self.state

        # if the input is an instance of State, get the inner merged data.
        if isinstance(state, State):
            state = state.merged_data

        # if the input state is a list of len(1)
        if isinstance(state, list) and len(state) == 1:
            state = state[0]

        self.value = self.model.predict(state, to_numpy=to_numpy, return_logits=True, set_output_data=False)
        return self.value

    #############
    # Operators #
    #############

    def __call__(self, state=None, to_numpy=False):
        """Predict the value."""
        return self.evaluate(state=state, to_numpy=to_numpy)

    def __copy__(self):
        """Return a shallow copy of the value approximator. This can be overridden in the child class."""
        return self.__class__(state=self.state, model=self.model)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the value approximator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        state = copy.deepcopy(self.state, memo) if isinstance(self.state, State) else copy.deepcopy(self.state)
        model = copy.deepcopy(self.model, memo)
        value = self.__class__(state=state, model=model)
        memo[self] = value
        return value


# alias
Value = ParametrizedValue


class ParametrizedQValue(QValueApproximator):  # ParametrizedValue, QValueApproximator):
    r"""Parametrized Q-Value Function Approximator (which accepts as inputs the states and actions)

    This value function approximator predicts the state-action value :math:`Q_{\phi}(s,a)`. Two different kind of
    state-action value function approximators can be defined:
    - where the action is given as input to the value function approximator. The action can be continuous or discrete.
    - where the action is given as output to the value function approximator. The output of such model is the
        state-action value for each DISCRETE action.

    By default, this class implements the value function approximator :math:`Q_{\phi}(s,a)` where the states and
    actions are given as inputs.
    """

    def __init__(self, state, action, model):
        """
        Initialize the parametrized state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): input / output action.
            model (Approximator): value function approximator.
        """
        # ParametrizedValue.__init__(self, state, model)
        # QValueApproximator.__init__(self, state, action)
        super(ParametrizedQValue, self).__init__(state, action)
        self.model = model

    ##############
    # Properties #
    ##############

    @property
    def model(self):
        """Return the model instance."""
        return self._model

    @model.setter
    def model(self, model):
        """Set the model / approximator instance."""
        if not isinstance(model, Approximator):
            raise TypeError("Expecting the model to be an instance of `Approximator`, instead got: "
                            "{}".format(type(model)))
        self._model = model

    @property
    def input_size(self):
        """Return the policy input size."""
        return self.model.input_size

    @property
    def output_size(self):
        """Return the policy output size."""
        return self.model.output_size

    @property
    def input_shape(self):
        """Return the policy input shape."""
        return self.model.input_shape

    @property
    def output_shape(self):
        """Return the policy output shape."""
        return self.model.output_shape

    @property
    def input_dim(self):
        """Return the input dimension of the policy; i.e. len(input_shape)."""
        return self.model.input_dim

    @property
    def output_dim(self):
        """Return the output dimension of the policy; i.e. len(output_shape)."""
        return self.model.output_dim

    @property
    def num_parameters(self):
        """Return the total number of parameters"""
        return self.model.num_parameters

    ###########
    # Methods #
    ###########

    def parameters(self):
        """
        Return an iterator over the learning model parameters.
        """
        return self.model.parameters()

    def named_parameters(self):
        """
        Return an iterator over the learning model parameters; yielding both the name and the parameter itself.
        """
        return self.model.named_parameters()

    def list_parameters(self):
        """
        Return the learning model parameters.
        """
        return self.model.list_parameters()

    def hyperparameters(self):
        """
        Return an iterator over the learning model hyper-parameters.
        """
        return self.model.hyperparameters()

    def named_hyperparameters(self):
        """
        Return an iterator over the learning model hyper-parameters; yielding both the name and the hyper-parameter
        itself.
        """
        return self.model.named_hyperparameters()

    def list_hyperparameters(self):
        """
        Return the learning model hyper-parameters
        """
        return self.model.list_hyperparameters()

    def get_vectorized_parameters(self, to_numpy=True):
        """
        Get the parameters in a vectorized form.

        Args:
            to_numpy (bool): if True, it will convert the 1D parameter vector into a numpy array.
        """
        return self.model.get_vectorized_parameters(to_numpy=to_numpy)

    def set_vectorized_parameters(self, vector):
        """
        Set the vectorized parameters.

        Args:
            np.array, torch.Tensor: 1D parameter vector.
        """
        self.model.set_vectorized_parameters(vector=vector)

    def evaluate(self, state=None, action=None, to_numpy=False):
        """Compute the output of the value function.

        Args:
            state (None, State, (list of) np.array, (list of) torch.Tensor): state input data. If None, it will get
                the data from the states that were given at the initialization.
            action (None, Action, (list of) np.array, (list of) torch.Tensor): input actions. If None, it will get
                the data from the actions that were given at the initialization.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
        """
        # if no input is given, take the provided inputs at the beginning
        if state is None:
            state = self.state
        if action is None:
            action = self.action

        # if the input is an instance of State, get the inner merged data.
        if isinstance(state, State):
            state = state.merged_data
        # if the input state is a list of len(1)
        if isinstance(state, list) and len(state) == 1:
            state = state[0]

        # if the input is an insrtance of Action, get the inner merged data.
        if isinstance(action, Action):
            action = action.merged_data
        # if the input action is a list of len(1)
        if isinstance(action, list) and len(action) == 1:
            action = action[0]

        self.value = self.model.predict([state, action], to_numpy=to_numpy, return_logits=True, set_output_data=False)
        return self.value

    #############
    # Operators #
    #############

    def __call__(self, state=None, action=None, to_numpy=False):
        """Predict the value."""
        return self.evaluate(state=state, action=action, to_numpy=to_numpy)

    def __copy__(self):
        """Return a shallow copy of the value approximator. This can be overridden in the child class."""
        return self.__class__(state=self.state, action=self.action, model=self.model)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the value approximator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        state = copy.deepcopy(self.state, memo) if isinstance(self.state, State) else copy.deepcopy(self.state)
        action = copy.deepcopy(self.action, memo) if isinstance(self.action, Action) else copy.deepcopy(self.action)
        model = copy.deepcopy(self.model, memo)
        value = self.__class__(state=state, action=action, model=model)
        memo[self] = value
        return value


# alias
QValue = ParametrizedQValue


class ParametrizedQValueOutput(ParametrizedQValue):
    r"""Parametrized Q-Value Function Approximator (which accepts as inputs the states and outputs a Q-value for each
    discrete action)

    This value function approximator predicts the state-action value :math:`Q_{\phi}(s,a)` for each discrete action.
    """

    def __init__(self, state, action, model):
        """
        Initialize the parametrized state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): output DISCRETE state.
            model (Approximator): value function approximator.
        """
        super(ParametrizedQValueOutput, self).__init__(state, action, model)

    ##############
    # Properties #
    ##############

    @property
    def action(self):
        """Return the action instance."""
        return self._action

    @action.setter
    def action(self, action):
        """Set the action input."""
        if isinstance(action, (int, float)):
            action = np.array([action])
        elif isinstance(action, Action):
            if not action.is_discrete():
                raise ValueError("The given actions are not discrete: {}".format(action))
        elif not isinstance(action, (torch.Tensor, np.ndarray)):
            raise TypeError("Expecting the action to be an int, float, Action, torch.Tensor, or np.ndarray.")
        self._action = action

    ###########
    # Methods #
    ###########

    def evaluate(self, state=None, action=None, to_numpy=False):
        """Compute the output of the value function.

        Args:
            state (None, State, (list of) np.array, (list of) torch.Tensor): state input data. If None, it will get
                the data from the states that were given at the initialization.
            action (None): this argument is discarded.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
        """
        # if no input is given, take the provided inputs at the beginning
        if state is None:
            state = self.state

        # if the input is an instance of State, get the inner merged data.
        if isinstance(state, State):
            state = state.merged_data
            if len(state) == 1:
                state = state[0]

        self.value = self.model.predict(state, to_numpy=to_numpy, return_logits=True, set_output_data=False)
        return self.value

    #############
    # Operators #
    #############

    def __call__(self, state=None, action=None, to_numpy=False):
        """Predict the value."""
        return self.evaluate(state=state, to_numpy=to_numpy)

    def __copy__(self):
        """Return a shallow copy of the value approximator. This can be overridden in the child class."""
        return self.__class__(state=self.state, action=self.action, model=self.model)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the value approximator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        state = copy.deepcopy(self.state, memo) if isinstance(self.state, State) else copy.deepcopy(self.state)
        action = copy.deepcopy(self.action, memo) if isinstance(self.action, Action) else copy.deepcopy(self.action)
        model = copy.deepcopy(self.model, memo)
        value = self.__class__(state=state, action=action, model=model)
        memo[self] = value
        return value


# alias
QValueOutput = ParametrizedQValueOutput
