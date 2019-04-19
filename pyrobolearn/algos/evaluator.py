#!/usr/bin/env python
"""Provide the Evaluator class used in the second step of RL algorithms

The evaluator assesses the quality of the actions/trajectories performed by the policy using the given returns.
It is the step performed after the exploration phase, and before the update step.
"""

from pyrobolearn.returns import Estimator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
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
        if not None and not isinstance(estimator, Estimator):
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
            verbose (bool): If true, print information on the standard output.
        """
        if self.estimator is not None:
            if verbose:
                print("\n#### Starting the Evaluation phase ####")

            # compute the returns
            returns = self.estimator.evaluate(self.storage)

            if verbose:
                # print("Returns: {}".format(returns))
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
