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

from pyrobolearn.storages import Storage

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Sampler(object):

    @property
    def batch_size(self):
        """Return the batch size."""
        return 0


class RandomSampler(Sampler):
    pass


class StorageSampler(Sampler):
    r"""Storage sampler

    Sampler used with the storage.
    """

    def __init__(self, storage, sampler=None, num_batches=10, batch_size=None, batch_size_bounds=None,
                 replacement=True, verbose=0):
        """
        Initialize the storage sampler.

        Args:
            storage (Storage): storage sampler.
            sampler (Sampler, None): If None, it will use a sampler that randomly sample batches of the storage. It
                will by default sample :attr:`num_batches`.
            num_batches (int): number of batches.
            batch_size (int, None): size of the batch. If None, it will be computed based on the size of the storage,
                where batch_size = size(storage) // num_batches. Note that the batch size must be smaller than the size
                of the storage itself. The num_batches * batch_size can however be bigger than the storage size if
                :attr:`replacement = True`.
            batch_size_bounds (tuple of int, None): if :attr:`batch_size` is None, we can instead specify the lower
                and upper bounds for the `batch_size`. For instance, we can set it to `(16, 128)` along with
                `batch_size=None`; this will result to compute `batch_size = size(storage) // num_batches` but if this
                one is too small (<16), it will be set to 16, and if this one is too big (>128), it will be set to 128.
            replacement (bool): if we should sample each element only one time, or we can sample the same ones multiple
                times.
            verbose (int, bool): verbose level, select between {0, 1, 2}. If 0=False, it won't print anything. If
                1=True, it will print basic information about the sampler. If verbose=2, it will print detailed
                information.
        """
        # set the storage
        self.storage = storage

        # set variables
        self._num_batches = num_batches
        self._replacement = bool(replacement)
        self._batch_size_bounds = batch_size_bounds
        self._batch_size_given = batch_size is not None
        self._verbose = verbose

        # set the sampler
        if sampler is None:

            # check batch size and compute it if necessary
            if batch_size is None:
                batch_size = self.size // num_batches

            # check batch size bounds
            if isinstance(batch_size_bounds, (tuple, list)) and len(batch_size_bounds) == 2:
                if batch_size < batch_size_bounds[0]:
                    batch_size = batch_size_bounds[0]
                elif batch_size > batch_size_bounds[1]:
                    batch_size = batch_size_bounds[1]

            # check the batch size * number of batches wrt the storage size
            if batch_size * num_batches > self.size and not self.replacement:
                raise ValueError("Expecting the batch size (={}) * num_batches (={}) to be smaller than the size of "
                                 "the storage (={}), if we can not use replacement.".format(batch_size, num_batches,
                                                                                            self.size))

            # subsampler
            if replacement:
                subsampler = torch_sampler.RandomSampler(data_source=range(self.size), replacement=replacement,
                                                         num_samples=self.size)
            else:
                subsampler = torch_sampler.SubsetRandomSampler(indices=range(self.size))

            # create sampler
            sampler = torch_sampler.BatchSampler(sampler=subsampler, batch_size=batch_size, drop_last=True)

        self.sampler = sampler

        if verbose:
            print("\nCreating sampler with size: {} - num batches: {} - batch size: {}".format(self.size, num_batches,
                                                                                               self.batch_size))

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
        if not isinstance(storage, Storage):
            raise TypeError("Expecting the storage to be an instance of `Storage`, instead got: "
                            "{}".format(type(storage)))
        self._storage = storage

    @property
    def size(self):
        """Return the total size of the storage number of time steps * number of processes."""
        return self.storage.size

    @property
    def filled_size(self):
        """Return the filled size of the storage."""
        return self.storage.filled_size

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
    def batch_size(self, batch_size):
        """Set the batch size."""
        # check the batch size * number of batches wrt the storage size
        if batch_size * self._num_batches > self.size and not self.replacement:
            raise ValueError("Expecting the batch size (={}) * num_batches (={}) to be smaller than the size of "
                             "the storage (={}), if we can not use replacement.".format(batch_size, self._num_batches,
                                                                                        self.size))

        # check batch size bounds
        if isinstance(self._batch_size_bounds, (tuple, list)) and len(self._batch_size_bounds) == 2:
            if batch_size < self._batch_size_bounds[0]:
                batch_size = self._batch_size_bounds[0]
            elif batch_size > self._batch_size_bounds[1]:
                batch_size = self._batch_size_bounds[1]

        # set the batch size for the sampler
        self.sampler.batch_size = batch_size

    @property
    def num_batches(self):
        """Return the number of batches (based on the size of the storage and the batch size)."""
        # return self.filled_size // self.batch_size
        return self._num_batches

    @num_batches.setter
    def num_batches(self, num_batches):
        """Set the number of batches."""
        self._num_batches = num_batches
        if not self._batch_size_given:
            self.batch_size = self.filled_size // num_batches
        else:
            if self.batch_size * self._num_batches > self.size and not self.replacement:
                raise ValueError("Expecting the batch size (={}) * num_batches (={}) to be smaller than the size of "
                                 "the storage (={}), if we can not use replacement.".format(self.batch_size,
                                                                                            self._num_batches,
                                                                                            self.size))

    @property
    def replacement(self):
        """Return the replacement boolean."""
        return self._replacement

    @property
    def batch_size_bounds(self):
        """Return the batch size bounds."""
        return self._batch_size_bounds

    ###########
    # Methods #
    ###########

    def generator(self, num_batches):
        """Return a generator.

        Args:
            num_batches (int): number of batches
        """
        # get filled size of the storage
        size = self.filled_size

        # get batch size
        self.batch_size = size // num_batches

        # check if there is a sub-sampler
        if hasattr(self.sampler, 'sampler'):
            if hasattr(self.sampler.sampler, 'data_source'):
                self.sampler.sampler.data_source = range(size)
            elif hasattr(self.sampler.sampler, 'indices'):
                self.sampler.sampler.indices = range(size)
        else:
            if hasattr(self.sampler, 'data_source'):
                self.sampler.data_source = range(size)
            elif hasattr(self.sampler, 'indices'):
                self.sampler.indices = range(size)

        for indices in self.sampler:
            yield self.storage.get_batch(indices)

    def __iter__(self):
        """Iterate over the storage."""
        # get the filled size
        size = self.filled_size

        if self._verbose:
            print("Storage filled size: {} - size: {}".format(size, self.size))

        # modify the sampler (by changing the size)
        # check if there is a sub-sampler
        if hasattr(self.sampler, 'sampler'):

            # change the size of the sampler
            if hasattr(self.sampler.sampler, 'data_source'):
                self.sampler.sampler.data_source = range(size)
            elif hasattr(self.sampler.sampler, 'indices'):
                self.sampler.sampler.indices = range(size)

            # compute the batch size if specified
            if not self._batch_size_given:
                self.batch_size = size // self.num_batches

        else:
            # change the size of the sampler
            if hasattr(self.sampler, 'data_source'):
                self.sampler.data_source = range(size)
            elif hasattr(self.sampler, 'indices'):
                self.sampler.indices = range(size)

        if self._verbose:
            print("\nCreating sampler with size: {} - num batches: {} - batch size: {}".format(size, self.num_batches,
                                                                                               self.batch_size))

        # provide the batches
        batch_idx = 0
        while True:  # this is to account for replacement = True
            for indices in self.sampler:
                batch_idx += 1
                yield self.storage.get_batch(indices)
                if batch_idx >= self.num_batches:
                    break

            if batch_idx >= self.num_batches:
                break


class BatchRandomSampler(StorageSampler):
    r"""Batch Random Sampler

    """

    def __init__(self, storage, num_batches=10, batch_size=None, batch_size_bounds=None, replacement=True, verbose=0):
        """
        Initialize the storage sampler.

        Args:
            storage (Storage): storage to sample the batches from.
            num_batches (int): number of batches.
            batch_size (int, None): size of the batch. If None, it will be computed based on the size of the storage,
                where batch_size = size(storage) // num_batches. Note that the batch size must be smaller than the size
                of the storage itself. The num_batches * batch_size can however be bigger than the storage size if
                :attr:`replacement = True`.
            batch_size_bounds (tuple of int, None): if :attr:`batch_size` is None, we can instead specify the lower
                and upper bounds for the `batch_size`. For instance, we can set it to `(16, 128)` along with
                `batch_size=None`; this will result to compute `batch_size = size(storage) // num_batches` but if this
                one is too small (<16), it will be set to 16, and if this one is too big (>128), it will be set to 128.
            replacement (bool): if we should sample each element only one time, or we can sample the same ones multiple
                times.
            verbose (int, bool): verbose level, select between {0, 1, 2}. If 0=False, it won't print anything. If
                1=True, it will print basic information about the sampler. If verbose=2, it will print detailed
                information.
        """
        super(BatchRandomSampler, self).__init__(storage=storage, num_batches=num_batches, batch_size=batch_size,
                                                 batch_size_bounds=batch_size_bounds, replacement=replacement,
                                                 verbose=verbose)
