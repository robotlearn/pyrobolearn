#!/usr/bin/env python
"""Provide the Updater class used in the third and final step of RL algorithms

The updater update the approximator (such as the policy and/or value function) parameters based on the loss, and
using the specified optmizer.

Dependencies:
- `pyrobolearn/approximators`: models (which contain parameters to update)
- `pyrobolearn/losses`: to compute the loss
- `pyrobolearn/optimizers`: the optimizers used to update the model parameters
- `pyrobolearn/samplers`:
"""

import collections

# TODO: makes the 5 following classes inherit from the same Parent class
from pyrobolearn.approximators import Approximator
from pyrobolearn.policies import Policy
from pyrobolearn.values import Value
from pyrobolearn.dynamics import DynamicModel
from pyrobolearn.actorcritics import ActorCritic
from pyrobolearn.exploration import Exploration  # TODO change that name to Explorer instead

from pyrobolearn.losses import Loss
from pyrobolearn.optimizers import Optimizer
from pyrobolearn.samplers import StorageSampler
from pyrobolearn.returns import Return, Target, Evaluator
from pyrobolearn.parameters.updater import ParameterUpdater


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Updater(object):
    r"""Updater

    (Model-free) reinforcement learning algorithms requires 3 steps:
    1. Explore: Explore and collect samples in the environment using the policy. The samples are stored in the
                given memory/storage unit.
    2. Evaluate: Assess the quality of the actions/trajectories using the returns.
    3. Update: Update the policy (and/or value function) parameters based on the loss

    This class focuses on the third step of RL algorithms.
    """

    def __init__(self, approximators, sampler, losses, optimizers, evaluators=None, updaters=None, ticks=None):
        """
        Initialize the update phase.

        Args:
            approximators (list of Policy, Value, ActorCritic,...): approximators to update based on the given losses.
            sampler (StorageSampler): sampler associated with the storage.
            losses (Loss, list/dict of losses): losses. If dict: key=approximator, value=loss.
            optimizers (Optimizer, or list/dict of optimizers): optimizer to use. If dict: key=approximator,
                value=optimizer.
            evaluators (list of Target, Return, Evaluator): list of sub-evaluators that are evaluated on batches at
                each update step before evaluating the losses and updating the parameters of the approximators. They
                modify the `current` attribute of the batch.
            updaters (None, dictionary, list of tuple): list of parameter updaters to run at the end.
            ticks (None, dictionary): dictionary containing as the key (updater or loss) and the value are the number
                of time steps to wait before updating the corresponding key. By default, it will evaluate the given
                losses and updaters at each time step.
        """
        self.approximators = approximators
        self.sampler = sampler
        self.losses = losses
        self.optimizers = optimizers
        self.evaluators = evaluators
        self.updaters = updaters
        self.ticks = ticks

        # counter
        self._cnt = 0

    ##############
    # Properties #
    ##############

    @property
    def approximators(self):
        """Return the list of approximators to update."""
        return self._approximators

    @approximators.setter
    def approximators(self, approximators):
        """Set the approximator instances."""
        if not isinstance(approximators, collections.Iterable):
            approximators = [approximators]
        for approximator in approximators:
            if not isinstance(approximator, (Approximator, Policy, Value, ActorCritic, DynamicModel, Exploration)):
                raise TypeError("Expecting the approximator to be an instance of `Approximator`, `Policy`, `Value`, "
                                "`ActorCritic`, `DynamicModel`, or `Exploration`. Instead got: "
                                "{}".format(type(approximator)))
        self._approximators = approximators

    @property
    def sampler(self):
        """Return the sampler instance."""
        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        """Set the sampler."""
        if not isinstance(sampler, StorageSampler):
            raise TypeError("Expecting the sampler to be an instance of `StorageSampler`, instead got: "
                            "{}".format(type(sampler)))
        self._sampler = sampler
        # self.storage = self._sampler.storage

    @property
    def storage(self):
        """Return the storage unit."""
        # return self._storage
        return self.sampler.storage

    @storage.setter
    def storage(self, storage):
        """Set the storage unit."""
        self.sampler.storage = storage

    @property
    def losses(self):
        """Return the losses (one for each approximator)."""
        return self._losses

    @losses.setter
    def losses(self, losses):
        """Set the losses."""
        # check that the losses are the correct data type
        if not isinstance(losses, collections.Iterable):
            losses = [losses]
        for loss in losses:
            if not isinstance(loss, Loss):
                raise TypeError("Expecting the loss to be an instance of `Loss`, instead got: {}".format(type(loss)))

        # check that the number of losses matches the number of approximators
        if len(losses) != len(self.approximators):
            raise ValueError("The number of losses does not match up with the number of approximators to update.")

        # set the losses
        self._losses = losses

    @property
    def optimizers(self):
        """Return the optimizers used to optimize the parameters of the approximators."""
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers):
        """Set the optimizers."""
        # check that the optimizers are the correct data type
        if not isinstance(optimizers, collections.Iterable):
            optimizers = [optimizers]
        for optimizer in optimizers:
            if not isinstance(optimizer, Optimizer):
                raise TypeError("Expecting optimizer to be an instance of `Optimizer`, instead got: "
                                "{}".format(type(optimizer)))

        # check that the number of optimizers match the number of approximators / losses
        if len(optimizers) != len(self.approximators):
            if len(optimizers) == 1:
                optimizers = optimizers * len(self.approximators)
            else:
                raise ValueError("Expecting the number of optimizers (={}) to match up with the number of "
                                 "approximators / losses (={})".format(len(optimizers), len(self.approximators)))

        # set the optimizers
        self._optimizers = optimizers

    @property
    def evaluators(self):
        """Return the (sub-)evaluator instances which are applied on batches at each update step before evaluating the
        losses and updating the parameters of the approximators."""
        return self._evaluators

    @evaluators.setter
    def evaluators(self, evaluators):
        """Set the (sub-)evaluators which are applied on each batch at each update step before evaluating the
        losses and updating the parameters of the approximators."""
        if evaluators is None:
            evaluators = []
        if not isinstance(evaluators, collections.Iterable):
            evaluators = [evaluators]
        for i, evaluator in enumerate(evaluators):
            if not isinstance(evaluator, (Target, Return, Evaluator)):
                raise TypeError("Expecting the {}th given 'evaluator' to be an instance of `Target`, `Return`, "
                                "`Evaluator`, instead got: {}".format(i, type(evaluator)))
        self._evaluators = evaluators

    @property
    def updaters(self):
        """Return the list of updaters (i.e. functions that updates some parameters and that are run at the end of an
        update step."""
        return self._updaters

    @updaters.setter
    def updaters(self, updaters):
        if updaters is None:
            updaters = []
        if not isinstance(updaters, collections.Iterable):
            updaters = [updaters]
        for i, updater in enumerate(updaters):
            if not isinstance(updater, ParameterUpdater):
                raise TypeError("Expecting the {}th updater to be an instance of `ParameterUpdater`, instead got: "
                                "{}".format(i, type(updater)))
        self._updaters = updaters

    @property
    def ticks(self):
        """Return the ticks."""
        return self._ticks

    @ticks.setter
    def ticks(self, ticks):
        """Set the ticks for each loss and updater."""
        # check type of the given ticks
        if ticks is None:
            ticks = dict()
        if not isinstance(ticks, dict):
            raise TypeError("Expecting the given ticks to be a dictionary, instead got: {}".format(type(ticks)))

        # check first the items already present in the ticks
        for key, value in ticks.iteritems():
            # check that the key is a Loss or ParamaterUpdater
            if not isinstance(key, (Loss, ParameterUpdater)):
                raise TypeError("Expecting the given key for the tick to be an instance of `Loss` or "
                                "`ParameterUpdater`, instead got: {}".format(type(key)))

            # check that the tick value is an int
            if not isinstance(value, int):
                if isinstance(value, float):
                    value = int(value)
                else:
                    raise TypeError("Expecting the given value for the tick to be an int, instead got: "
                                    "{}".format(type(value)))

            # check that the tick is bigger than 0
            if value <= 0:
                raise ValueError("Expecting the given value for the tick to be an integer bigger than 0, instead got: "
                                 "{}".format(value))

        # set the tick for each loss
        for loss in self.losses:
            if loss not in ticks:
                ticks[loss] = 1

        # set the tick for each updater
        for updater in self.updaters:
            if updater not in ticks:
                ticks[updater] = 1

        # set the ticks
        self._ticks = ticks

    ###########
    # Methods #
    ###########

    def update(self, num_epochs=1, num_batches=10, verbose=False):
        """
        Update the given approximators (policies, value functions, etc).

        Args:
            num_epochs (int): number of epochs.
            num_batches (int): number of batches.
            verbose (bool): If true, print information on the standard output.

        Returns:
            dict: dictionary of losses. There is a key for each loss, and the value is a nested list which contains
                the obtained loss for each epoch and for each batch in the corresponding epoch.
        """
        # set the number of batches
        self.sampler.num_batches = num_batches

        # keep history of each loss
        losses = {}

        if verbose:
            print("\n#### 3. Starting the Update phase ####")

        # for each epoch
        for epoch in range(num_epochs):

            # for each batch
            for batch_idx, batch in enumerate(self.sampler):

                if verbose:
                    print("Epoch: {}/{} - Batch: {}/{} with size {}".format(epoch + 1, num_epochs, batch_idx + 1,
                                                                            num_batches, batch.size))

                # evaluate the evaluators with the current parameters on the given batch and save the results in the
                # batch's `current` attribute
                for evaluator in self.evaluators:
                    if verbose:
                        print("Subevaluation on the batch using the estimator: {}".format(evaluator))
                    evaluator.evaluate(batch, store=True)

                # update each approximator based on the loss on which it is evaluated and using the specified optimizer
                for approximator, loss, optimizer in zip(self.approximators, self.losses, self.optimizers):

                    # if time to update
                    if self._cnt % self.ticks[loss] == 0:

                        if verbose:
                            print("\t Compute loss: {}".format(loss))

                        # compute loss on the data (the loss knows what to do with the batch)
                        loss_value = loss.compute(batch)

                        # append the loss value in the history of losses
                        if loss not in losses:
                            losses[loss] = [[] for epoch in range(num_epochs)]
                        losses[loss][epoch].append(loss_value.detach())

                        # update parameters
                        if verbose:
                            print("\t Optimize the parameters for {} using the loss value: {}".format(approximator,
                                                                                                      loss_value))
                        optimizer.optimize(approximator.parameters(), loss_value)

                    # call each updater
                    for updater in self.updaters:
                        if self._cnt % self.ticks[updater] == 0:
                            if verbose:
                                print("\tRun updater {}".format(updater))
                            updater()

                # increase counter
                self._cnt += 1

        if verbose:
            print("Losses: {}".format(losses))
            print("#### End of the Update phase ####")

        return losses  # shape=(epochs, batches)

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a representation string."""
        return self.__class__.__name__

    def __str__(self):
        """Return a string describing the class."""
        return self.__class__.__name__

    def __call__(self, num_epochs=1, num_batches=10, verbose=False):
        """
        Update the given approximators (policies, value functions, etc).

        Args:
            num_epochs (int): number of epochs.
            num_batches (int): number of batches.
            verbose (bool): If true, print information on the standard output.

        Returns:
            dict: dictionary of losses. There is a key for each loss, and the value is a nested list which contains
                the obtained loss for each epoch and for each batch in the corresponding epoch.
        """
        self.update(num_epochs=num_epochs, num_batches=num_batches, verbose=verbose)
