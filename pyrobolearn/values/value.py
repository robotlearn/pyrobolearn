#!/usr/bin/env python
"""Provides the various abstract value function approximators.

It can use learning models as most of these models learn a function (that is how to map inputs to outputs).
For instance, a value network is a function represented by a neural network that maps a state to a value (real number).

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
- `pyrobolearn.approximators` (and thus `pyrobolearn.models`)
"""

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

    def compute(self, *args, **kwargs):
        """Predict the value."""
        pass

    def __call__(self, *args, **kwargs):
        """Predict the value."""
        return self.compute(*args, **kwargs)


class StateValueApproximator(ValueApproximator):
    r"""State Value Approximator

    Compute :math:`V(s)`.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state):
        super(StateValueApproximator, self).__init__(state)


class ActionValueApproximator(ValueApproximator):
    r"""Action Value Approximator

    Compute :math:`Q(s,a)`.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, actions):
        super(ActionValueApproximator, self).__init__(state)
        self.actions = actions


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

    def compute(self, state=None, to_numpy=True):
        """Compute the output of the value function.

        Args:
            state (None, State, (list of) np.array, (list of) torch.Tensor): state input data. If None, it will get
                the data from the inputs that were given at the initialization.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
        """
        self.value = self.model.predict(state, to_numpy=to_numpy, return_logits=True, set_output_data=False)
        return self.value

    def __call__(self, state=None, to_numpy=True):
        """Predict the value."""
        return self.compute(state=state, to_numpy=to_numpy)


class ParametrizedStateActionValue(ParametrizedValue):
    r"""Parametrized State Action Value Function Approximator

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
        super(ParametrizedStateActionValue, self).__init__(state, model)
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


class ParametrizedStateOutputActionValue(ParametrizedValue):
    r"""Parametrized State Output Action Value Function Approximator

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
        super(ParametrizedStateOutputActionValue, self).__init__(state, model)
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
        elif isinstance(action, Action):
            if not action.is_discrete():
                raise ValueError("The given actions are not discrete: {}".format(action))
        elif not isinstance(action, (torch.Tensor, np.ndarray)):
            raise TypeError("Expecting the action to be an int, float, Action, torch.Tensor, or np.ndarray.")
        self._action = action
