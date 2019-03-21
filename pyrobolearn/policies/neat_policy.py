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
__license__ = "MIT"
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

    def __init__(self, states, actions, num_hidden=0, activation_fct='relu', network_type='feedforward',
                 aggregation='sum', weights_limits=(-20, 20), bias_limits=(-20, 20), rate=1, *args, **kwargs):
        r"""Initialize the neural network policy for the NEAT algorithm.
        """
        model = NEATApproximator(states, actions, num_hidden=num_hidden, activation_fct=activation_fct,
                                 network_type=network_type, aggregation=aggregation, weights_limits=weights_limits,
                                 bias_limits=bias_limits)
        super(NEATPolicy, self).__init__(states, actions, model, rate=rate, *args, **kwargs)

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

    def update_config(self, config):
        self.model.update_config(config)

    def set_network(self, genome=None, config=None):
        self.model.set_network(genome, config)

    def act(self, state, deterministic=True):
        if (self.cnt % self.rate) == 0:
            self.last_action = self.model.predict(state)
        self.cnt += 1
        return self.last_action

    def sample(self, state):
        pass


class NEATFeedForwardPolicy(NEATPolicy):
    r"""NEAT feed-forward policy

    This creates a feed-forward network policy.
    """

    def __init__(self, states, actions, genome):
        super(NEATFeedForwardPolicy, self).__init__(states, actions, genome, network_type='feedforward')


class NEATRecurrentPolicy(NEATPolicy):
    r"""NEAT recurrent policy

    This creates a recurrent network policy.
    """

    def __init__(self, states, actions, genome):
        super(NEATRecurrentPolicy, self).__init__(states, actions, genome, network_type='recurrent')
