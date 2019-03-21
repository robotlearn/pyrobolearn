#!/usr/bin/env python
"""Define the NEAT model class.

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
    import neat
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install NEAT directly via 'pip install neat-python'.")

# from pyrobolearn.models import Model

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NEATModel(object):  # Model):
    r"""NEAT Model

    NEAT stands for "Neuro-Evolution through Augmenting Topologies" [1] and allows the evolution of not only the
    parameters/weights but also the topological structure of neural networks. The model (i.e. the neural network)
    is tightly coupled with the algorithm that modifies it. By the structure of the neural network, we mean
    the type (i.e. forward or recurrent) and number of connection, as well as the type (i.e. using non-linearity
    activation function) and number of nodes can change.

    This model works in a Reinforcement Learning setting, where exploration is carried out in the parameter and
    hyper-parameter spaces of the neural network.

    Warnings: The associated algorithm is a little bit special and currently only works with the corresponding
    learning model.

    References:
        [1] "Evolving Neural Networks through Augmenting Topologies", Stanley et al., 2002
        [2] NEAT-Python
            - documentation: https://neat-python.readthedocs.io/en/latest/index.html
            - github repo: https://github.com/CodeReclaimers/neat-python
        [3] PyTorch NEAT (built upon NEAT-Python): https://github.com/uber-research/PyTorch-NEAT
    """

    def __init__(self, num_inputs, num_outputs, num_hidden=0, activation_fct='relu', network_type='feedforward',
                 aggregation='sum', weights_limits=(-20, 20), bias_limits=(-20, 20)):
        # super(NEATModel, self).__init__()

        if network_type != 'feedforward' and network_type != 'recurrent':
            raise ValueError("Expecting the 'network_type' argument to be 'feedforward' or 'recurrent'. Received "
                             "instead {}".format('network_type'))
        self.network_type = network_type

        # set config file
        # more info about genome's config file: https://neat-python.readthedocs.io/en/latest/config_file.html
        # more info about activation fct: https://neat-python.readthedocs.io/en/latest/activation.html
        self.config_dict = {'[NEAT]': {'fitness_criterion': 'max',
                                       'pop_size': 10,
                                       'fitness_threshold': 100,
                                       'no_fitness_termination': True,
                                       'reset_on_extinction': True},
                            '[DefaultSpeciesSet]': {'compatibility_threshold': 3.0},
                            '[DefaultStagnation]': {'species_fitness_func': 'max',
                                                    'max_stagnation': 15,
                                                    'species_elitism': 2},
                            '[DefaultReproduction]': {'elitism': 2,
                                                      'survival_threshold': 0.2,
                                                      'min_species_size': 2},
                            '[DefaultGenome]': {'activation_default': activation_fct,
                                                'activation_mutate_rate': 0.0,
                                                'activation_options': activation_fct,
                                                # node aggregation options
                                                'aggregation_default': aggregation,
                                                'aggregation_mutate_rate': 0.0,
                                                'aggregation_options': aggregation,
                                                # node bias options
                                                'bias_init_mean': 0.0,
                                                'bias_init_stdev': 1.0,
                                                'bias_init_type': 'gaussian',
                                                'bias_max_value': bias_limits[1],
                                                'bias_min_value': bias_limits[0],
                                                'bias_mutate_power': 0.5,
                                                'bias_mutate_rate': 0.7,
                                                'bias_replace_rate': 0.1,
                                                # genome compatibility options
                                                'compatibility_disjoint_coefficient': 1.0,
                                                'compatibility_weight_coefficient': 0.5,
                                                # connection add/remove rates
                                                'conn_add_prob': 0.5,
                                                'conn_delete_prob': 0.5,
                                                # connection enable options
                                                'enabled_default': True,
                                                'enabled_mutate_rate': 0.01,
                                                'feed_forward': self.network_type == 'feedforward',
                                                'initial_connection': 'full_nodirect',
                                                # node add/remove rates
                                                'node_add_prob': 0.1,
                                                'node_delete_prob': 0.1,
                                                # network parameters
                                                'num_hidden': num_hidden,
                                                'num_inputs': num_inputs,
                                                'num_outputs': num_outputs,
                                                # node response options
                                                'response_init_mean': 1.0,
                                                'response_init_stdev': 0.0,
                                                'response_max_value': 30.0,
                                                'response_min_value': -30.0,
                                                'response_mutate_power': 0.0,
                                                'response_mutate_rate': 0.0,
                                                'response_replace_rate': 0.0,
                                                # connection weight options
                                                'weight_init_mean': 0.0,
                                                'weight_init_stdev': 1.0,
                                                'weight_max_value': weights_limits[1],
                                                'weight_min_value': weights_limits[0],
                                                'weight_mutate_power': 0.5,
                                                'weight_mutate_rate': 0.8,
                                                'weight_replace_rate': 0.1}}

        # create config
        config_file = self._create_config_file()
        self._config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                   neat.DefaultStagnation, config_file)

        # create initial population
        self.population = neat.Population(self.config)

        # set initial genome (first genome from the population)
        self._genome = self.population.population[1]

        # create network
        self.model = self.set_network(self.genome, self.config)

    ##############
    # Properties #
    ##############

    @property
    def input_dims(self):
        """Return the input dimension of the model"""
        return len(self.model.input_nodes)

    @property
    def output_dims(self):
        """Return the output dimension of the model"""
        return len(self.model.output_nodes)

    @property
    def input_shape(self):
        """Return the input shape of the model"""
        return tuple([self.input_dims])

    @property
    def output_shape(self):
        """Return the output shape of the model"""
        return tuple([self.output_dims])

    @property
    def config(self):
        """Return the config object"""
        return self._config

    @config.setter
    def config(self, config):
        """Set the config file (str) or object."""
        if not isinstance(config, neat.config.Config):
            raise TypeError("Expecting genome to be an instance of neat.config.Config.")
        self._config = config
        # create population
        self.population = neat.Population(self._config)

    @property
    def genome(self):
        return self._genome

    @genome.setter
    def genome(self, genome):
        if not isinstance(genome, neat.genome.DefaultGenome):
            raise TypeError("Expecting genome to be an instance of neat.genome.DefaultGenome type")
        self._genome = genome

        # create network
        self.model = self.set_network(self._genome, self.config)

    @property
    def network(self):
        return self.model

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def is_parametric():
        return True

    @staticmethod
    def is_linear():
        return False

    @staticmethod
    def is_recurrent():
        return True

    @staticmethod
    def is_probabilistic():
        return False

    @staticmethod
    def is_discriminative():
        return True

    @staticmethod
    def is_generative():
        return False

    @staticmethod
    def load(filename):
        """
        Load a model from memory.

        Args:
            filename (str): file that contains the model.
        """
        return pickle.load(open(filename, 'rb'))

    ###########
    # Methods #
    ###########

    def _create_config_str(self, config_dict=None):
        """Create string describing the config file from a config dictionary"""
        if config_dict is None:
            config_dict = self.config_dict

        # create config file
        config = []
        for section, parameters in config_dict.items():
            config.append(section)
            for key, value in parameters.items():
                config.append(key + ' = ' + str(value))
            config.append('')

        # return string describing the config file
        return '\n'.join(config)

    def _create_config_file(self, config_dict=None):
        """Create config file from a config dictionary"""
        config = self._create_config_str(config_dict)
        filename = 'config.txt'

        # create config file
        with open(filename, 'w') as f:
            f.write(config)

        # return path to the config file
        return filename

    def set_network(self, genome=None, config=None):
        # check arguments
        if genome is None:
            genome = self.genome
        if config is None:
            config = self.config

        # Create the neural network
        if self.network_type == 'feedforward':
            self.model = neat.nn.FeedForwardNetwork.create(genome, config)
        elif self.network_type == 'recurrent':
            self.model = neat.nn.RecurrentNetwork.create(genome, config)
        else:
            raise TypeError('Choose between feedforward or recurrent or implement your own type of NN.')

        return self.model

    def update_config(self, config):
        # update config (dict)
        if isinstance(config, dict):
            self.config_dict.update(config)
            config_file = self._create_config_file()
            self._config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                       neat.DefaultStagnation, config_file)
        elif isinstance(config, neat.Config):
            self._config = config

        # create new population
        self.population = neat.Population(self.config)

        # set new genome (first genome from the population)
        self._genome = self.population.population[1]

        # set new network
        self.model = self.set_network(self.genome, self.config)

    def parameters(self):
        """Returns an iterator over the model parameters."""
        return []

    def named_parameters(self):
        """Returns an iterator over the model parameters, yielding both the name and the parameter itself"""
        return []

    def hyperparameters(self):
        pass

    def get_params(self):
        pass

    def get_hyperparams(self):
        pass

    def predict(self, x=None):
        return self.model.activate(x)

    def save(self, filename):
        """
        Save the model in memory.

        Args:
            filename (str): file to save the model in.
        """
        pickle.dump(self, open(filename, 'wb'))
