# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the NEAT policy class.

This uses the Neuro-Evolution through Augmenting topologies (NEAT) framework. It allows the evolution of not only the
parameters/weights but also the topological structure of neural networks. Note that the model associated with
this policy (i.e. the neural network) is tightly coupled with the algorithm that modifies it.
"""

import numpy as np
try:
    import cPickle as pickle
except ImportError as e:
    import pickle

try:
    # from neat import nn, population, config, statistics
    import neat
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install NEAT directly via 'pip install neat-python'.")

from pyrobolearn.policies.policy import Policy
from pyrobolearn.approximators import NEATApproximator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NEATPolicy(Policy):
    r"""NEAT Policy

    NEAT stands for "Neuro-Evolution through Augmenting Topologies" [1] and allows the evolution of not only the
    parameters/weights but also the topological structure of neural networks. The model associated with this
    policy (i.e. the neural network) is tightly coupled with the algorithm that modifies it.
    By the structure of the neural network, we mean the type (i.e. forward or recurrent) and number of connection,
    as well as the type (i.e. using non-linearity activation function) and number of nodes can change.

    Exploration is thus carried out in the parameter and hyper-parameter spaces.

    Warnings: The associated algorithm is a little bit special and currently only works with the corresponding
    policy/learning model.

    References:
        [1] "Evolving Neural Networks through Augmenting Topologies", Stanley et al., 2002
        [2] NEAT-Python
            - documentation: https://neat-python.readthedocs.io/en/latest/index.html
            - github repo: https://github.com/CodeReclaimers/neat-python
        [3] PyTorch NEAT (built upon NEAT-Python): https://github.com/uber-research/PyTorch-NEAT
    """

    def __init__(self, state, action, num_hidden=0, activation_fct='relu', network_type='feedforward',
                 aggregation='sum', weights_limits=(-20, 20), bias_limits=(-20, 20), rate=1, preprocessors=None,
                 postprocessors=None, *args, **kwargs):
        r"""Initialize the neural network policy for the NEAT algorithm.

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
            num_hidden (int): number of units in the hidden layer
            activation_fct (str): activation function to use.
            network_type (str): type of neural network. Select between 'feedforward' and 'recurrent'.
            aggregation (str): how to aggregate the input signals of a node. Select between 'sum', 'product', 'max',
                'min', 'maxabs', 'median', and 'mean'.
            weights_limits (tuple): weight limits / bounds. The tuple contains the lower and upper bounds.
            bias_limits (tuple): bias limits / bounds. The tuple contains the lower and upper bounds.
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
                stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
                executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
        """
        model = NEATApproximator(state, action, num_hidden=num_hidden, activation_fct=activation_fct,
                                 network_type=network_type, aggregation=aggregation, weights_limits=weights_limits,
                                 bias_limits=bias_limits, preprocessors=preprocessors, postprocessors=postprocessors)
        super(NEATPolicy, self).__init__(state, action, model, rate=rate, *args, **kwargs)

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

    def update_config(self, config):
        """Update the configuration file."""
        self.model.update_config(config)

    def set_network(self, genome=None, config=None):
        """Set the genome network."""
        self.model.set_network(genome, config)

    # def act(self, state, deterministic=True):
    #     if (self.cnt % self.rate) == 0:
    #         self.last_action = self.model.predict(state)
    #     self.cnt += 1
    #     return self.last_action
    #
    # def sample(self, state):
    #     pass


# class NEATFeedForwardPolicy(NEATPolicy):
#     r"""NEAT feed-forward policy
#
#     This creates a feed-forward network policy.
#     """
#
#     def __init__(self, states, actions, genome):
#         super(NEATFeedForwardPolicy, self).__init__(states, actions, genome, network_type='feedforward')
#
#
# class NEATRecurrentPolicy(NEATPolicy):
#     r"""NEAT recurrent policy
#
#     This creates a recurrent network policy.
#     """
#
#     def __init__(self, states, actions, genome):
#         super(NEATRecurrentPolicy, self).__init__(states, actions, genome, network_type='recurrent')
