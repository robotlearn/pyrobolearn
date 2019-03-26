#!/usr/bin/env python
"""Provides the `transition`/`dynamic` function approximators in RL.

Dynamic models allows to compute the next state given the current state and action; that is,
:math:`s_{t+1} = f(s_t, a_t)` (if deterministic) or :math:`s_{t+1} \sim p(.| s_t, a_t)`.

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
- `pyrobolearn.approximators` (and thus `pyrobolearn.models`)
"""

from abc import ABCMeta, abstractmethod
import collections
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


class DynamicModel(object):
    r"""Dynamic/Transition Model

    In the reinforcement learning setting, the dynamic model is the transition probability function associated to the
    environment which provides the next state given the current state and action. The agent / policy has no control
    over it. However, it can often be learned from data samples acquired by interacting with the environment.
    This allows to model the environment and then perform internal simulations requiring thus less samples from the
    real environment.

    When a dynamic model is involved, this is known as "Model-based Reinforcement Learning", also known as "Optimal
    Control". These methods usually require less samples as they learn a model of the environment. This contrasts with
    model-free on-policy search algorithms that do not exploit the data collected in previous episodes, and thus
    require a lot of samples per episode. Note that off-policy methods while exploiting past data are particular to
    a specific task (due to the specific reward function).

    Dynamic models allows one also to plan. Additionally, using a *differentiable* dynamic model with a
    *differentiable* policy allows to unfold the consequences of selected actions for a certain time horizon, and
    to take the gradient of the loss with respect to the first action which can then be used to estimate a better
    action to undertake by taking a small step (using the gradient) towards an action that optimize better the loss.
    This is notably useful for *model predictive control* (MPC).

    Dynamic models can be deterministic :math:`s_{t+1} = f_{\varphi}(s_t, a_t)` or stochastic
    :math:`s_{t+1} \sim P_{\varphi}(s_{t+1} | s_t, a_t)`, where :math:`\varphi` is the possible set of parameters if
    the dynamic model is a trainable model, :math:`s_t` and :math:`a_t` are the current state and action respectively,
    and :math:`s_{t+1}` is the predicted next state.

    They are 2 main ways to build a dynamic model:
    1. build it using a mathematical model
        Pros: mathematical guarantees (such as stability), predictable, ...
        Cons: linearization, unmodeled phenomenon, assumptions that might be violated (rigid body), and so on
    2. learn it from data, by letting the policy interacts with the environment
        Pros: data-driven approach and thus more flexible and potentially more accurate.
        Cons: might require a lot of samples to be accurate, might overfit which could lead to a mismatch between
            the real and the learned dynamic model, often no guarantees and could be unpredictable.

    Note that learning a wrong dynamic model can have drastic consequences on the learned policy as this last one
    depends on the returned predicted states by the environment.

    References:
        [1] https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
        [2] "Optimal control theory: An introduction", Kirk, 2004
    """
    __metaclass__ = ABCMeta

    def __init__(self, states, actions, next_states=None):
        """
        Initialize the dynamic transition probability :math:`p(s_{t+1} | s_t, a_t)`, or dynamic transition function
        :math:`s_{t+1} = f(s_t, a_t)`.

        Args:
            states (State): state inputs.
            actions (Action): action inputs.
            next_states (State, None): state outputs. If None, it will take the state inputs as the outputs.
        """
        # set inputs
        self.states = states
        self.actions = actions

        # set outputs
        self.next_states = next_states

    ##############
    # Properties #
    ##############

    @property
    def states(self):
        """Return the state instance."""
        return self._states

    @states.setter
    def states(self, states):
        """Set the states."""
        if not isinstance(states, State):
            raise TypeError("Expecting the given states to be an instance of `State`, instead got: "
                            "{}".format(type(states)))
        self._states = states

    @property
    def actions(self):
        """Return thge action instance."""
        return self._actions

    @actions.setter
    def actions(self, actions):
        """Set the actions."""
        if not isinstance(actions, Action):
            raise TypeError("Expecting the given actions to be an instance of `Action`, instead got: "
                            "{}".format(type(actions)))
        self._actions = actions

    @property
    def next_states(self):
        """Return the next state instance."""
        return self._next_states

    @next_states.setter
    def next_states(self, states):
        """Set the next states."""
        if states is None:
            states = self.states
        elif not isinstance(states, State):
            raise TypeError("Expecting the given next_states to be an instance of `State`, instead got: "
                            "{}".format(type(states)))
        self._next_states = states

    ###########
    # Methods #
    ###########

    @abstractmethod
    def predict(self, states=None, actions=None, deterministic=False):
        """
        Predict the next state given the current state and action.

        Args:
            states (None, State, (list of) np.array, (list of) torch.Tensor): input states.
            actions (None, Action, (list of) np.array, (list of) torch.Tensor): input actions.

        Returns:
            (list of) np.array, (list of) torch.Tensor: predicted next state data.
        """
        pass

    def __call__(self, states, actions):
        """
        Return predicted next state given the current state and action.

        Args:
            states (None, State, (list of) np.array, (list of) torch.Tensor): input states.
            actions (None, Action, (list of) np.array, (list of) torch.Tensor): input actions.

        Returns:
            (list of) np.array, (list of) torch.Tensor: predicted next state data.
        """
        return self.predict(states, actions)


class ParametrizedDynamicModel(DynamicModel):
    r"""Learnable Parametrized Dynamic Model

    Dynamic model that can be trained.
    """

    def __init__(self, states, actions, model, next_states=None, distributions=None):
        """
        Initialize the dynamic transition probability :math:`p(s_{t+1} | s_t, a_t)`.

        Args:
            states (State): state inputs.
            actions (Action): action inputs.
            model (Approximator): approximator (inner learning model).
            next_states (State, None): state outputs. If None, it will take the state inputs as the outputs.
            distributions ((list of) torch.distributions.Distribution, None): distribution to use to sample the next
                state. If None, it will be deterministic.
        """
        # set inner model
        super(ParametrizedDynamicModel, self).__init__(states, actions, next_states)
        self.model = model
        self.distributions = distributions

    ##############
    # Properties #
    ##############

    @property
    def model(self):
        """Return the model instance."""
        return self._model

    @model.setter
    def model(self, model):
        """Set the approximator model."""
        if not isinstance(model, Approximator):
            raise TypeError("Expecting the model to be an instance of `Approximator`, instead got: "
                            "{}".format(type(model)))
        self._model = model

    @property
    def distributions(self):
        """Return the list of distributions."""
        return self._distributions

    @distributions.setter
    def distributions(self, distributions):
        if distributions is None:
            distributions = []
        elif isinstance(distributions, torch.distributions.Distribution):
            distributions = [distributions]

        if len(distributions) != 0 and len(distributions) != len(self.next_states):
            raise ValueError("Expecting the number of distributions (={}) to match the number of states (={})"
                             ".".format(len(distributions), len(self.next_states)))
        self._distributions = distributions

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

    def predict(self, states=None, actions=None, deterministic=False, to_numpy=True, set_next_state_data=True):
        """
        Predict the next state given the current state and action.

        Args:
            states (None, State, (list of) np.array, (list of) torch.Tensor): input states.
            actions (None, Action, (list of) np.array, (list of) torch.Tensor): input actions.
            deterministic (bool): if True, the prediction will be deterministic. If False and if a distribution was
                given at initialization, then the predicted output will be stochastic.
            to_numpy (bool): If True, it will return a (list of) np.array.
            set_next_state_data (bool): if True, it will set the next state data.

        Returns:
            (list of) np.array, (list of) torch.Tensor: predicted next state data.
        """
        # if no input is given, take the provided inputs at the beginning
        data = self.model.predict([states, actions], to_numpy=to_numpy, return_logits=True,
                                  set_output_data=False)

        # return predicted next state
        if deterministic or len(self.distributions) == 0:
            return data
        data = [distribution(datum).sample() for distribution, datum in zip(self.distributions, data)]
        if set_next_state_data:
            self.next_states.data = data
        return data


# class GPDynamicModel(ParametrizedDynamicModel):
#     r"""Gaussian Process Dynamic Model
#
#     Dynamic model using Gaussian Processes.
#
#     Pros: good from a mathematical point of view: integrate uncertainty on the dynamic model
#     Cons:
#
#     ..seealso: PILCO
#     """
#
#     def __init__(self, states, actions):
#         super(GPDynamicModel, self).__init__(states, actions)
