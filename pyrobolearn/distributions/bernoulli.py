#!/usr/bin/env python
"""Define the discrete Bernoulli distribution class.
"""

import torch
import numpy as np


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Bernoulli(torch.distributions.Bernoulli):
    r"""Bernoulli distribution

    Type: discrete, binary

    "The Bernoulli distribution is the discrete probability distribution of a random variable which takes the value 1
    with probability :math:`p` and the value 0 with probability :math:`q = 1-p`, that is, the probability distribution
    of any single experiment that asks a yes/no question; the question results in a boolean-valued outcome, a single
    bit of information whose value is success with probability :math:`p` and failure with probability :math:`q`." [1]

    References:
        [1] Bernoulli distribution: https://en.wikipedia.org/wiki/Bernoulli_distribution
    """

    def __init__(self, probs=None, logits=None):
        """
        Initialize the Bernoulli distribution on the given manifold.

        Args:
            probs (torch.Tensor, None): event probabilities module.
            logits (torch.Tensor, None): event logits module.
        """
        # call superclass
        super(Bernoulli, self).__init__(probs=probs, logits=logits)

    def mode(self):
        """Return the mode of the Bernoulli distribution."""
        return torch.gt(self.probs, 0.5).float()
