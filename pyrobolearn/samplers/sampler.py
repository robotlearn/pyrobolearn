#!/usr/bin/env python
"""Provide the Sampler class from which all sampling strategies inherit from.

Sampling strategies can be classified into two categories: probability or deterministic sampling methods.

Deterministic sampling methods include: convenience sampling, purposive sampling, quota sampling, and
referral/snowball sampling.

Probability sampling methods include: random sampling, stratified sampling, systematic sampling, cluster sampling,
multi-stage sampling, importance sampling, MCMC sampling, etc.

Others include maximum variation sampling, extreme case sampling, etc.

References:
    [1] https://towardsdatascience.com/sampling-techniques-a4e34111d808
    [2] https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/sampling-in-statistics/
    [3] https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html
    [4] https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html
"""

import torch.utils.data.sampler as torch_sampler
import torch.utils.data.dataset as torch_dataset

from pyrobolearn.storages import RolloutStorage

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Sampler(object):
    pass


class RandomSampler(Sampler):
    pass


class StorageSampler(Sampler):
    r"""Storage sampler

    Sampler used with the storage.
    """

    def __init__(self, storage, sampler=None, num_batches=10):
        """
        Initialize the storage sampler.

        Args:
            storage (RolloutStorage): rollout storage.
            sampler (Sampler, None): If None, it will use a sampler that randomly sample batches of the storage. It
                will by default sample :attr:`num_batches`.
            num_batches (int): number of batches
        """
        # set the storage
        self.storage = storage

        # set the sampler
        if sampler is None:
            batch_size = self.size // num_batches
            if batch_size > self.size:
                raise ValueError("Expecting the batch size (={}) to be smaller than the size of the storage (={})"
                                 ".".format(batch_size, self.size))
            sampler = torch_sampler.BatchSampler(sampler=torch_sampler.SubsetRandomSampler(range(self.size)),
                                                 batch_size=batch_size, drop_last=False)
        self.sampler = sampler

    ##############
    # Properties #
    ##############

    @property
    def storage(self):
        """Return the storage instance."""
        return self._storage

    @storage.setter
    def storage(self, storage):
        """Set the storage instance."""
        if not isinstance(storage, RolloutStorage):
            raise TypeError("Expecting the storage to be an instance of `RolloutStorage`, instead got: "
                            "{}".format(type(storage)))
        self._storage = storage

    @property
    def size(self):
        """Return the total size of the storage number of time steps * number of processes."""
        return self.storage.size

    @property
    def sampler(self):
        """Return the sampler."""
        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        """Set the sampler."""
        if not isinstance(sampler, (torch_sampler.Sampler, Sampler)):
            raise TypeError("Expecting the sampler to be an instance of `torch.utils.data.sampler` or `Sampler`, "
                            "instead got: {}".format(type(sampler)))
        self._sampler = sampler

    @property
    def batch_size(self):
        """Return the number of batches."""
        return self.sampler.batch_size

    @batch_size.setter
    def batch_size(self, size):
        """Set the batch size."""
        if size > self.size:
            raise ValueError("Expecting the batch size (={}) to be smaller than the size of the storage (={})"
                             ".".format(size, self.size))
        self.sampler.batch_size = size

    @property
    def num_batches(self):
        """Return the number of batches (based on the size of the storage and the batch size)."""
        return self.size // self.batch_size

    @num_batches.setter
    def num_batches(self, num_batches):
        """Set the number of batches."""
        self.batch_size = self.size // num_batches

    ###########
    # Methods #
    ###########

    def generator(self, num_batches):
        """Return a generator.

        Args:
            num_batches (int): number of batches
        """
        self.batch_size = self.size // num_batches

        for indices in self.sampler:
            yield self.storage.get_batch(indices)

    def __iter__(self):
        """Iterate over the storage."""
        for indices in self.sampler:
            yield self.storage.get_batch(indices)
