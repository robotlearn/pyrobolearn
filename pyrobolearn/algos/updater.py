#!/usr/bin/env python
"""Provide the Updater class used in the third and final step of RL algorithms

The updater update the approximator (such as the policy and/or value function) parameters based on the loss, and
using the specified optmizer.
"""

# TODO: makes the 5 following classes inherit from the same Parent class
from pyrobolearn.approximators import Approximator
from pyrobolearn.policies import Policy
from pyrobolearn.values import Value
from pyrobolearn.dynamics import DynamicModel
from pyrobolearn.actorcritics import ActorCritic
from pyrobolearn.exploration import Exploration  # TODO change that name to Explorer instead

from pyrobolearn.losses import Loss
from pyrobolearn.optimizers import Optimizer
from pyrobolearn.storages import Storage, RolloutStorage, Batch
from pyrobolearn.samplers import StorageSampler


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Updater(object):
    r"""Updater

    (Model-free) reinforcement learning algorithms requires 3 steps:
    1. Explore: Explore and collect samples in the environment using the policy. The samples are stored in the
                given memory/storage unit.
    2. Evaluate: Assess the quality of the actions/trajectories using the estimators.
    3. Update: Update the policy (and/or value function) parameters based on the loss

    This class focuses on the third step of RL algorithms.
    """

    def __init__(self, approximators, sampler, losses, optimizers):
        """
        Initialize the update phase.

        Args:
            approximators (list of Policy, Value, ActorCritic,...): approximators to update based on the given losses.
            sampler (StorageSampler): sampler associated with the storage.
            losses (Loss, list/dict of losses): losses. If dict: key=approximator, value=loss.
            optimizers (Optimizer, or list/dict of optimizers): optimizer to use. If dict: key=approximator,
                value=optimizer.
        """
        self.approximators = approximators
        self.sampler = sampler
        self.losses = losses
        self.optimizers = optimizers
        self._evaluator = ApproximatorEvaluator(approximators)

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
        if not isinstance(approximators, list):
            approximators = [approximators]
        for approximator in approximators:
            if not isinstance(approximator, (Approximator, Policy, Value, ActorCritic, DynamicModel, Exploration)):
                raise TypeError("Expecting the approximator to be an instance of `Approximator`, `Policy`, `Value`, "
                                "`ActorCritic`, `DynamicModel`, or `Exploration`. Instead got: "
                                "{}".format(type(approximator)))
        self._approximators = approximators
        self._evaluator = ApproximatorEvaluator(self._approximators)

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
        # if not isinstance(storage, RolloutStorage):
        #     raise TypeError("Expecting the storage to be an instance of `Storage`, instead got: "
        #                     "{}".format(type(storage)))
        # self._storage = storage

    @property
    def losses(self):
        """Return the losses (one for each approximator)."""
        return self._losses

    @losses.setter
    def losses(self, losses):
        """Set the losses."""
        # check that the losses are the correct data type
        if not isinstance(losses, list):
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
        if not isinstance(optimizers, list):
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
    def evaluator(self):
        """Return the evaluator instance which evaluates the approximators on the given batch.."""
        return self._evaluator

    ###########
    # Methods #
    ###########

    def update(self, num_batches=10):
        """
        Update the given approximators (policies, value functions, etc).

        Args:
            num_batches (int): number of batches

        Returns:
            list: list of losses
        """
        # set the number of batches
        self.sampler.num_batches = num_batches

        # for each batch
        for batch in self.sampler:

            # evaluation with the current parameters
            self.evaluator.evaluate(batch)

            # update each approximator based on the loss on which it is evaluated and using the specified optimizer
            for approximator, loss, optimizer in zip(self.approximators, self.losses, self.optimizers):

                # compute loss on the data (the loss knows what to do)
                loss = loss.compute(batch)

                # update parameters
                optimizer.optimize(approximator.parameters(), loss)

        return self.losses

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a representation string."""
        return self.__class__.__name__

    def __str__(self):
        """Return a string describing the class."""
        return self.__class__.__name__

    def __call__(self, num_batches=10):  # , storage, losses):
        """Update the approximators."""
        self.update(num_batches=num_batches)


class ApproximatorEvaluator(object):
    r"""Approximators evaluator

    Approximators evaluator used mostly during the update phase. Evaluate the various approximators on the given batch.

    This consists:
    - for policies, to compute :math:`\pi_{\theta}(a|s)` and :math:`\pi_{\theta}(.|s)` if possible.
    - for value functions, to compute :math:`V_{\phi}(s)`, :math:`Q_{\phi}(s,a)`, and/or :math:`A_{\phi}`(s,a)
    - for dynamic models, to compute :math:``
    """

    def __init__(self, approximators):
        """
        Initialize the evaluator for the approximators.

        Args:
            approximators ((list of) Approximator): approximators
        """
        self.approximators = approximators

    ##############
    # Properties #
    ##############

    @property
    def approximators(self):
        """Return the list of approximators to update."""
        return self._approximators

    @approximators.setter
    def approximators(self, approximators):
        """Set the list of approximators to update."""
        if not isinstance(approximators, list):
            approximators = [approximators]
        for approximator in approximators:
            if not isinstance(approximator, (Approximator, Policy, Value, ActorCritic, DynamicModel, Exploration)):
                raise TypeError("Expecting the approximator to be an instance of `Approximator`, `Policy`, `Value`, "
                                "`ActorCritic`, `DynamicModel`, or `Exploration`. Instead got: "
                                "{}".format(type(approximator)))
        self._approximators = approximators

    ###########
    # Methods #
    ###########

    def evaluate(self, batch):
        """Evaluate the various approximators."""
        if not isinstance(batch, Batch):
            raise TypeError("Expecting the given batch storage to be an instance of `Batch`, instead got: "
                            "{}".format(type(batch)))
        # sub-evaluation with the current parameter
        for approximator in self.approximators:
            if isinstance(approximator, (Policy, Exploration)):
                actions, action_distributions = approximator.evaluate(batch['observations'])
                batch.current['actions'] = actions
                batch.current['action_distributions'] = action_distributions
            elif isinstance(approximator, Value):
                values = approximator.evaluate(batch['observations'], batch['actions'])
                batch.current['values'] = values
            elif isinstance(approximator, ActorCritic):
                actions, action_distributions, values = approximator.evaluate(batch['observations'], batch['actions'])
                batch.current['actions'] = actions
                batch.current['action_distributions'] = action_distributions
                batch.current['values'] = values
            elif isinstance(approximator, DynamicModel):
                next_states, state_distributions = approximator.evaluate(batch['observations'], batch['actions'])
                batch.current['next_states'] = next_states
                batch.current['state_distributions'] = state_distributions
            else:
                raise TypeError("Expecting the approximator to be an instance of `Policy`, `Value`, `ActorCritic`, or "
                                "`DynamicModel`, instead got: {}".format(type(approximator)))
        return batch


class PolicyEvaluator(object):
    r"""Policy evaluator

    Evaluate a policy by computing :math:`\pi_{\theta}(a|s)` and if possible the distribution :math:`\pi(.|s)`. The
    policy is evaluated on a batch.
    """

    def __init__(self, policy, batch=None):
        """Initialize the policy evaluator.

        policy (Policy): policy to evaluate.
        batch (None, Batch): initial batch.
        """
        self.policy = policy
        self.batch = batch

    ##############
    # Properties #
    ##############

    @property
    def policy(self):
        """Return the policy instance."""
        return self._policy

    @policy.setter
    def policy(self, policy):
        """Set the policy."""
        if not isinstance(policy, Policy):
            raise TypeError("Expecting the given policy to be an instance of `Policy`, instead got: "
                            "{}".format(type(policy)))
        self._policy = policy

    ###########
    # Methods #
    ###########

    def evaluate(self, batch=None):
        """Evaluate the policy on the given batch. If None, it will evaluate on the previous batch."""
        # check batch
        if batch is None:
            batch = self.batch
            if batch is None:
                raise ValueError("Expecting a batch to be given.")

        # evaluate policy
        actions, action_distributions = self.policy.evaluate(batch['observations'])

        # put them in the batch
        batch.current['actions'] = actions
        batch.current['action_distributions'] = action_distributions

        # return batch
        return batch


class ValueEvaluator(object):
    r"""Value evaluator

    Evaluate a value by computing :math:`V_{\phi}(s)`, :math:`Q_{\phi}(s,a)`, and / or :math:`A_{\phi}(s,a)`.
    The value is evaluated on a batch.
    """

    def __init__(self, value, batch=None):
        """Initialize the value evaluator.

        value (Value): value to evaluate.
        batch (None, Batch): initial batch.
        """
        self.value = value
        self.batch = batch

    ##############
    # Properties #
    ##############

    @property
    def value(self):
        """Return the value instance."""
        return self._value

    @value.setter
    def value(self, value):
        """Set the value function approximator."""
        if not isinstance(value, Value):
            raise TypeError("Expecting the given value to be an instance of `Value`, instead got: "
                            "{}".format(type(value)))
        self._value = value

    ###########
    # Methods #
    ###########

    def evaluate(self, batch=None):
        """Evaluate the value on the given batch. If None, it will evaluate on the previous batch."""
        # check batch
        if batch is None:
            batch = self.batch
            if batch is None:
                raise ValueError("Expecting a batch to be given.")

        # evaluate value
        values = self.value.evaluate(batch['observations'], batch['actions'])

        # put them in the batch
        batch.current['values'] = values

        # return batch
        return batch


class ActorCriticEvaluator(object):
    r"""ActorCritic evaluator

    Evaluate an action by computing :math:`\pi_{\theta}(a|s)` and if possible the distribution :math:`\pi(.|s)`. It
    also evaluates the value by computing :math:`V_{\phi}(s)`, :math:`Q_{\phi}(s,a)`, and / or :math:`A_{\phi}(s,a)`.
    Both are evaluated on a batch.
    """

    def __init__(self, actorcritic, batch=None):
        """Initialize the actorcritic evaluator.

        actorcritic (ActorCritic): actorcritic to evaluate.
        batch (None, Batch): initial batch.
        """
        self.actorcritic = actorcritic
        self.batch = batch

    ##############
    # Properties #
    ##############

    @property
    def actorcritic(self):
        """Return the actor-critic instance."""
        return self._actorcritic

    @actorcritic.setter
    def actorcritic(self, actorcritic):
        """Set the actor-critic."""
        if not isinstance(actorcritic, ActorCritic):
            raise TypeError("Expecting the given actorcritic to be an instance of `ActorCritic`, instead got: "
                            "{}".format(type(actorcritic)))
        self._actorcritic = actorcritic

    ###########
    # Methods #
    ###########

    def evaluate(self, batch=None):
        """Evaluate the actorcritic on the given batch. If None, it will evaluate on the previous batch."""
        # check batch
        if batch is None:
            batch = self.batch
            if batch is None:
                raise ValueError("Expecting a batch to be given.")

        # evaluate actorcritic
        actions, action_distributions, values = self.actorcritic.evaluate(batch['observations'])  # , batch['actions'])

        # put them in the batch
        batch.current['actions'] = actions
        batch.current['action_distributions'] = action_distributions
        batch.current['values'] = values

        # return batch
        return batch


class DynamicModelEvaluator(object):
    r"""Dynamic model evaluator

    Evaluate the next state given the current state and action.
    """

    def __init__(self, dynamic_model, batch=None):
        """Initialize the dynamic_model evaluator.

        dynamic_model (ActorCritic): dynamic_model to evaluate.
        batch (None, Batch): initial batch.
        """
        self.dynamic_model = dynamic_model
        self.batch = batch

    ##############
    # Properties #
    ##############

    @property
    def dynamic_model(self):
        """Return the dynamic_model instance."""
        return self._dynamic_model

    @dynamic_model.setter
    def dynamic_model(self, dynamic_model):
        """Set the dynamic model."""
        if not isinstance(dynamic_model, DynamicModel):
            raise TypeError("Expecting the given dynamic_model to be an instance of `ActorCritic`, instead got: "
                            "{}".format(type(dynamic_model)))
        self._dynamic_model = dynamic_model

    ###########
    # Methods #
    ###########

    def evaluate(self, batch=None):
        """Evaluate the dynamic_model on the given batch. If None, it will evaluate on the previous batch."""
        # check batch
        if batch is None:
            batch = self.batch
            if batch is None:
                raise ValueError("Expecting a batch to be given.")

        # evaluate dynamic_model
        next_states, state_distributions = self.dynamic_model.evaluate(batch['observations'], batch['actions'])

        # put them in the batch
        batch.current['next_states'] = next_states
        batch.current['state_distributions'] = state_distributions

        # return batch
        return batch
