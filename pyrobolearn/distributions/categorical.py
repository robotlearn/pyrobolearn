#!/usr/bin/env python
"""Define the discrete Categorical distribution class.
"""

import torch

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
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

    @staticmethod
    def from_list(categoricals):
        """
        Convert a list of Categorical [cat1, cat2, ..., catN] to a single Categorical distribution with N logits /
        probs.

        Args:
            categoricals (list of Categorical): list of Categorical distributions.

        Returns:
            Categorical: resulting single Categorical distribution.
        """
        return Categorical(probs=torch.stack([categorical.probs for categorical in categoricals]))

    def __getitem__(self, indices):
        """
        Get the corresponding Categoricals from the single Categorical distribution. That is, if the single Categorical
        distribution has multiple logits / probs, it selects the corresponding Categorical distributions from it.

        Examples:
            categorical = Categorical(probs=torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.7, 0.3]]))
            categorical[[0,2]]  # this returns Categorical(probs=torch.tensor([[0.25, 0.75], [0.7, 0.3]]))
            categorical[:2]  # this returns Categorical(probs=torch.tensor([[0.25, 0.75], [0.6, 0.4]]))

        Args:
            indices (int, list of int, slices): indices.

        Returns:
            Categorical: resulting sliced Categorical distribution.
        """
        return Categorical(probs=self.probs[indices])
