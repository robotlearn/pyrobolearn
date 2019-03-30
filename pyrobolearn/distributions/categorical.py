#!/usr/bin/env python
"""Define the discrete Categorical distribution class.
"""

import torch

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Categorical(torch.distributions.Categorical):
    r"""Categorical distribution

    Type: discrete, multiple categories

    The Categorical module accepts as inputs the discrete logits or probabilities modules, and returns the categorical
    distribution (that inherits from `torch.distributions.Categorical`).

    Description: "A categorical distribution (also called a generalized Bernoulli distribution, multinoulli
    distribution) is a discrete probability distribution that describes the possible results of a random variable that
    can take on one of K possible categories, with the probability of each category separately specified." [1]

    References:
        [1] Categorical distribution: https://en.wikipedia.org/wiki/Categorical_distribution
    """

    def __init__(self, probs=None, logits=None):
        """
        Initialize the Categorical distribution on the given manifold.

        Args:
            probs (torch.Tensor, None): event probabilities module.
            logits (torch.Tensor, None): event logits module.
        """
        # call superclass
        super(Categorical, self).__init__(probs=probs, logits=logits)

    @property
    def mode(self):
        """Return the mode of the Categorical distribution."""
        return self.probs.argmax(dim=-1, keepdim=True)
