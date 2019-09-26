# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the Neural Network (NN) Policies.

Define the various neural network policies that can be used.
"""

from pyrobolearn.approximators import NNApproximator, MLPApproximator
from pyrobolearn.policies.policy import Policy


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NNPolicy(Policy):
    r"""Neural Network Policy

    Defines the neural network policy.
    """

    def __init__(self, state, action, model=None, rate=1, preprocessors=None, postprocessors=None, *args, **kwargs):
        """
        Initialize the Neural network policy.

        Args:
            action (Action): At each step, by calling `policy.act(state)`, the `action` is computed by the policy,
                and can be given to the environment. As with the `state`, the type and size/shape of each inner
                action can be inferred and could be used to automatically build a policy. The `action` connects the
                policy with a controllable object (such as a robot) in the environment.
            state (State): By giving the `state` to the policy, it can automatically infer the type and size/shape
                of each inner state, and thus can be used to automatically build a policy. At each step, the `state`
                is filled by the environment, and read by the policy. The `state` connects the policy with one or
                several objects (including robots) in the environment. Note that some policies don't use any state
                information.
            model (NN, NNApproximator): NN model
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
                stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
                executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
            *args (list): list of arguments (this is not used in this class).
            **kwargs (dict): dictionary of arguments (this is not used in this class).
        """
        if model is None:
            raise ValueError("Expecting a NN model for the NN policy")
        else:
            # checking the input dimension of the model and the dimension of states
            # checking the output dimension of the model and the dimension of actions
            pass

        super(NNPolicy, self).__init__(state, action, model, rate=rate, preprocessors=preprocessors,
                                       postprocessors=postprocessors, *args, **kwargs)

    # def act(self, state, deterministic=True):
    #     pass
    #
    # def sample(self, state):
    #     pass


class MLPPolicy(NNPolicy):
    r"""Multi-Layer Perceptron (MLP) Policy

    Defines a MLP policy, which is a feedforward fully-connected neural network with linear layers and nonlinear
    activation functions.
    """

    def __init__(self, state, action, hidden_units=(), activation='linear', last_activation=None,
                 dropout=None, rate=1, preprocessors=None, postprocessors=None):
        """Initialize MLP policy.

        Args:
            state (State): 1D-states that is feed to the policy (the input dimensions will be inferred from the
                            states)
            action (Action): 1D-actions outputted by the policy and will be applied in the simulator (the output
                              dimensions will be inferred from the actions)
            hidden_units (list/tuple of int): number of hidden units in the corresponding layer
            activation (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the
            last_activation (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout (None, float, or list/tuple of float/None): dropout probability.
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
                stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
                executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
        """
        model = MLPApproximator(state, action, hidden_units=hidden_units,
                                activation=activation, last_activation=last_activation,
                                dropout=dropout, preprocessors=preprocessors, postprocessors=postprocessors)
        super(MLPPolicy, self).__init__(state, action, model, rate=rate)

    # def act(self, state, deterministic=True):
    #     return self.model.predict(state)
