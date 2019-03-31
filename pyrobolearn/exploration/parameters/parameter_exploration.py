#!/usr/bin/env python
"""Provide the parameter exploration strategies.

Parameter exploration is used in reinforcement learning algorithms, and describes how the policy explores in the
environment. Note that the policy is the only (probability) function that we have control over; we do not control the
dynamic transition (probability) function nor the reward function. In parameter exploration, we explore the parameter
space of the policy.

Note that parameter exploration is an episode-based exploration strategy where the parameters of the policy are only
perturbed at the beginning of an episode, and unchanged during that particular episode.

Parameter exploration might change a bit the structure of the policy while running.

References:
    [1] "Evolution strategies as a scalable alternative to reinforcement learning", Salimans et al., 2017
    [2] "Parameter Space Noise for Exploration", Plappert et al., 2018
"""

import torch

from pyrobolearn.exploration import Exploration


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ParameterExploration(Exploration):
    r"""Parameter Exploration (aka Episode-based RL)

    Explore in the parameter space of the policy. At each episode, we apply a small variation on the parameters
    of the policy and keep it fixed during the whole episode (hence the name 'episode-based' RL).

    Assume a policy is denoted by :math:`\pi_{\theta}(a|s)` which maps states :math:`s` to action :math`a`, and
    is parametrized by :math:`\theta` which are the parameters that can be learned/optimized/trained. In parameter
    space exploration, the parameters :math:`\theta` are sampled from a probability distribution, such as a
    Gaussian distribution such that :math:`\theta \sim \mathcal{N}(\theta_k, \Sigma)`.

    This way of exploring is notably used in:
    - population-based algorithms such as evolutionary algorithms (ES, NEAT). Note that in `NEAT`, the topology
    of the learning model (i.e. neural network) is also explored along with its weights.
    - reinforcement learning algorithms like PoWER, and others.

    Pros:
    - when sampling several parameters (and thus policies), each one of them can be evaluated in a parallel manner.
    These policies are thus independent.
    - giving the same state to a policy results in the same action (in contrast to action exploration).

    References:
        [1] "Parameter Space Noise for Exploration", Plappert et al., 2018
    """

    def __init__(self, policy):
        super(ParameterExploration, self).__init__(policy)
        self._parameters = policy.get_vectorized_parameters(to_numpy=False)

    ##############
    # Properties #
    ##############

    @property
    def parameters(self):
        """Returns the parameters."""
        return self._parameters

    @property
    def size(self):
        """Returns the dimension of the parameters."""
        return self.parameters.size(-1)

    ###########
    # Methods #
    ###########

    def reset(self):
        # sample new set of parameters for policy
        pass

    def act(self, state=None, deterministic=True, to_numpy=True, return_logits=False, apply_action=True):
        """Perform the action given the state."""
        actions = self.policy.act(state)


# class ModelUncertaintyExploration(Exploration):
#     r"""Model Uncertainty Exploration
#
#     Exploration based on the uncertainty of a learned dynamic model.
#
#     References:
#     [1]
#     """
#
#     def __init__(self, policy):
#         super(ModelUncertaintyExploration, self).__init__(policy)
