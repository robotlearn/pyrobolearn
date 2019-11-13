#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Evaluator class used in the second step of RL algorithms

The evaluator assesses the quality of the actions/trajectories performed by the policy using the given returns.
It is the step performed after the exploration phase, and before the update step.
"""

import torch
from pyrobolearn.returns import Estimator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Evaluator(object):
    r"""Evaluator

    (Model-free) reinforcement learning algorithms requires 3 steps:
    1. Explore: Explore and collect samples in the environment using the policy. The samples are stored in the
                given memory/storage unit.
    2. Evaluate: Assess the quality of the actions/trajectories using the returns.
    3. Update: Update the policy (and/or value function) parameters based on the loss

    This class focuses on the second step of RL algorithms.

    Note that step is used in the on-policy case, where we evaluate complete trajectories based on estimators
    """

    def __init__(self, estimator):
        """
        Initialize the Evaluation phase.

        Args:
            estimator (Estimator, None): estimator used to evaluate the actions performed by the policy.
        """
        self.estimator = estimator

    ##############
    # Properties #
    ##############

    @property
    def estimator(self):
        """Return the estimator used to evaluate the policy."""
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        """Set the estimator."""
        if estimator is not None and not isinstance(estimator, Estimator):
            raise TypeError("Expecting estimator to be an instance of `Estimator` or None, instead got: "
                            "{}".format(type(estimator)))
        self._estimator = estimator

    @property
    def storage(self):
        """Return the storage unit."""
        return self.estimator.storage

    @storage.setter
    def storage(self, storage):
        """Set the storage unit."""
        self.estimator.storage = storage

    ###########
    # Methods #
    ###########

    def evaluate(self, verbose=False):
        """
        Evaluate the trajectories performed by the policy.

        Args:
            verbose (int, bool): verbose level, select between {0=False, 1=True, 2}. If 1 or 2, it will print
                information about the evaluation process. The level 2 will print more detailed information. Do not use
                it when the states / actions are big or high dimensional, as it could be very hard to make sense of
                the data.
        """
        if self.estimator is not None:
            if verbose:
                print("\n#### 2. Starting the Evaluation phase ####")
                print("Using estimator: {}".format(self.estimator))

            # compute the returns
            returns = self.estimator.evaluate(self.storage)

            if verbose > 1:
                # print("Returns: {}".format(returns))

                print("\nFinal storage status: ")
                states = self.storage['states'][0]
                num_step, num_traj = states.shape[:2]
                states = states.view(-1, *states.size()[2:])
                print("states: {}".format(torch.cat((torch.Tensor(list(range(num_step)) * num_traj).view(-1, 1),
                                                     states), dim=1)))
                actions = self.storage['actions'][0]
                actions = actions.view(-1, *actions.size()[2:])
                print("actions: {}".format(torch.cat((torch.Tensor(list(range(num_step - 1)) * num_traj).view(-1, 1),
                                                      actions), dim=1)))
                rewards = self.storage['rewards'][:, :, 0]
                print("rewards: {}".format(torch.cat((torch.arange(len(rewards), dtype=torch.float).view(-1, 1),
                                                      rewards), dim=1)))
                masks = self.storage['masks'][:, :, 0]
                print("masks: {}".format(torch.cat((torch.arange(len(masks), dtype=torch.float).view(-1, 1),
                                                    masks), dim=1)))
                returns = self.storage[self.estimator][:, :, 0]
                print("returns: {}".format(torch.cat((torch.arange(len(returns), dtype=torch.float).view(-1, 1),
                                                      returns), dim=1)))

                print("\n#### End of the Evaluation phase ####")

            elif verbose:
                print("#### End of the Evaluation phase ####")

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return the representation string."""
        return self.__class__.__name__

    def __str__(self):
        """Return the class string."""
        return self.__class__.__name__

    def __call__(self, verbose=False):
        """
        Evaluate the trajectories performed by the policy.

        Args:
            verbose (bool): If true, print information on the standard output.
        """
        self.evaluate(verbose=verbose)
