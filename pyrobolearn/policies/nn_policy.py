#!/usr/bin/env python
"""Define the Neural Network (NN) Policies.

Define the various neural network policies that can be used.
"""

from pyrobolearn.approximators import NNApproximator, MLPApproximator
from pyrobolearn.policies.policy import Policy


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NNPolicy(Policy):
    r"""Neural Network Policy

    Defines the neural network policy. If the model is not given,

    Examples:
        simulator = Bullet()
        robot = Robot(simulator)
        policy = NNPolicy(Robot, states=['joint_positions', 'joint_velocities'], actions=['joint_positions'])
    """

    def __init__(self, states, actions, model=None, *args, **kwargs):
        if model is None:
            raise ValueError("Expecting a NN model for the NN policy")
        else:
            # checking the input dimension of the model and the dimension of states
            # checking the output dimension of the model and the dimension of actions
            pass

        super(NNPolicy, self).__init__(states, actions, model, *args, **kwargs)

    def act(self, state, deterministic=True):
        pass

    def sample(self, state):
        pass


class MLPPolicy(NNPolicy):
    r"""Multi-Layer Perceptron (MLP) Policy

    Defines a MLP policy, which is a feedforward fully-connected neural network with linear layers and nonlinear
    activation functions.
    """

    def __init__(self, states, actions, hidden_units=(),
                 activation_fct='linear', last_activation_fct=None, dropout_prob=None,
                 preprocessors=None, postprocessors=None):
        """Initialize MLP policy.

        Args:
            states (State): 1D-states that is feed to the policy (the input dimensions will be inferred from the
                            states)
            actions (Action): 1D-actions outputted by the policy and will be applied in the simulator (the output
                              dimensions will be inferred from the actions)
            hidden_units (list/tuple of int): number of hidden units in the corresponding layer
            activation_fct (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the
            last_activation_fct (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout_prob (None, float, or list/tuple of float/None): dropout probability.
        """
        model = MLPApproximator(states, actions, hidden_units=hidden_units,
                                activation_fct=activation_fct, last_activation_fct=last_activation_fct,
                                dropout_prob=dropout_prob, preprocessors=preprocessors, postprocessors=postprocessors)
        super(MLPPolicy, self).__init__(states, actions, model)

    def act(self, state, deterministic=True):
        return self.model.predict(state)
