#!/usr/bin/env python
"""Computes the various targets based on value function on batches of trajectories/transitions (used in RL).

The approximators are evaluated in . targets that are evaluated are placed inside the given batch, which can then be accessed by other classes.

Dependencies:
- `pyrobolearn.storages`
- `pyrobolearn.values`
"""

from pyrobolearn.approximators import Approximator
from pyrobolearn.policies import Policy
from pyrobolearn.values import Value
from pyrobolearn.dynamics import DynamicModel
from pyrobolearn.actorcritics import ActorCritic
from pyrobolearn.exploration import Exploration  # TODO change that name to Explorer instead

from pyrobolearn.storages import Batch
from pyrobolearn.returns import Return, Estimator


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

    Evaluator on the batches.
    """

    def _evaluate(self, batch):
        """Compute/evaluate the evaluator on the given batch, and return the result.

        Args:
            batch (Batch): batch containing the transitions / trajectories.

        Returns:
            torch.Tensor: evaluated targets.
        """
        raise NotImplementedError

    def evaluate(self, batch, store=True):
        """Compute/evaluate the evaluator on the given batch and insert the result in the given batch.

        Args:
            batch (Batch): batch containing the transitions / trajectories.
            store (bool): If True, it will save the evaluation of the target in the given batch.

        Returns:
            torch.Tensor: evaluated targets.
        """
        # check batch type
        if not isinstance(batch, Batch):
            raise TypeError("Expecting the given 'batch' to be an instance of `Batch`, instead got: "
                            "{}".format(type(batch)))

        output = self._evaluate(batch)
        if store:  # store the target in the batch if specified
            if isinstance(output, list):
                # outputs = []
                for key, value in output:
                    batch.current[key] = value
                #     outputs.append(value)
                # output = outputs
            else:
                batch.current[self] = output
        return output

    def __call__(self, batch, store=True):
        """Evaluate the evaluator on the given batch."""
        return self.evaluate(batch)


class AdvantageEvaluator(Evaluator):
    r"""Advantage evaluator

    Compute :math:`\hat{A}_t = R_t - V(s_t)`.
    """

    def __init__(self, returns, value, standardize=False):
        r"""
        Initialize the advantage evaluator.

        Args:
            returns (Return, Estimator): returns.
            value (Value): value function.
            standardize (bool): if True, it will standardize the advantage.
        """
        # check returns
        if not isinstance(returns, (Return, Estimator)):
            raise TypeError("Expecting the given 'returns' to be an instance of `Return`, `Estimator`, instead got: "
                            "{}".format(returns))
        self._returns = returns

        # check value function
        if not isinstance(value, Value):
            raise TypeError("Expecting the given 'value' to be an instance of `Value`, instead got: "
                            "{}".format(type(value)))
        self._value = value

        # set standardize
        self._standardize = bool(standardize)

    def _evaluate(self, batch):
        """
        Evaluate the advantage estimate.

        Args:
            batch (Batch): batch containing the transitions / trajectories.

        Returns:
            torch.Tensor: advantage estimates.
        """
        # check value
        if self._value in batch.current:
            values = batch.current[self._value]
        else:
            values = self._value(batch['states'])

        # check returns
        if self._returns in batch.current:
            returns = batch.current[self._returns]
        elif self._returns in batch:
            returns = batch[self._returns]
        else:
            returns = self._returns(batch)

        # compute advantage estimates
        advantages = returns - values

        # standardize the advantage estimates
        if self._standardize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.e-5)

        return advantages


class PolicyEvaluator(Evaluator):
    r"""Policy evaluator

    Evaluate a policy by computing :math:`\pi_{\theta}(a|s)` and if possible the distribution :math:`\pi(.|s)`. The
    policy is evaluated on a batch.
    """

    def __init__(self, policy):
        """Initialize the policy evaluator.

        policy (Exploration): policy (with exploration) to evaluate.
        """
        # check policy
        if not isinstance(policy, Exploration):
            raise TypeError("Expecting the given policy to be an instance of `Exploration`, instead got: "
                            "{}".format(type(policy)))
        self._policy = policy

    def _evaluate(self, batch):
        """Evaluate the policy on the given batch. If None, it will evaluate on the previous batch.

        Args:
            batch (Batch): batch containing the transitions / trajectories.

        Returns:
            torch.Tensor: advantage estimates.
        """
        # evaluate policy
        actions, action_distributions = self._policy.predict(batch['states'])

        # return actions and distribution over actions
        return [('actions', actions), ('action_distributions', action_distributions)]


class ValueEvaluator(Evaluator):
    r"""Value evaluator

    Evaluate a value by computing :math:`V_{\phi}(s)`, :math:`Q_{\phi}(s,a)`, and / or :math:`A_{\phi}(s,a)`.
    The value is evaluated on a batch.
    """

    def __init__(self, value):
        """Initialize the value evaluator.

        value (Value): value to evaluate.
        """
        if not isinstance(value, Value):
            raise TypeError("Expecting the given value to be an instance of `Value`, instead got: "
                            "{}".format(type(value)))
        self._value = value

    def _evaluate(self, batch):
        """Evaluate the value on the given batch. If None, it will evaluate on the previous batch.

        Args:
            batch (Batch): batch containing the transitions / trajectories.
        """
        # evaluate value
        values = self._value.evaluate(batch['states'], batch['actions'])

        # return values
        return [(self._value, values)]


# class ActorCriticEvaluator(Evaluator):
#     r"""ActorCritic evaluator
#
#     Evaluate an action by computing :math:`\pi_{\theta}(a|s)` and if possible the distribution :math:`\pi(.|s)`. It
#     also evaluates the value by computing :math:`V_{\phi}(s)`, :math:`Q_{\phi}(s,a)`, and / or :math:`A_{\phi}(s,a)`.
#     Both are evaluated on a batch.
#     """
#
#     def __init__(self, actorcritic):
#         """Initialize the actorcritic evaluator.
#
#         actorcritic (ActorCritic): actorcritic to evaluate.
#         """
#         if not isinstance(actorcritic, ActorCritic):
#             raise TypeError("Expecting the given actorcritic to be an instance of `ActorCritic`, instead got: "
#                             "{}".format(type(actorcritic)))
#         self._actorcritic = actorcritic
#
#     def _evaluate(self, batch):
#         """Evaluate the actorcritic on the given batch. If None, it will evaluate on the previous batch.
#
#         Args:
#             batch (Batch): batch containing the transitions / trajectories.
#         """
#         # evaluate actorcritic
#         actions, action_distributions, values = self._actorcritic.evaluate(batch['states'])  # , batch['actions'])
#
#         # put them in the batch
#         batch.current['actions'] = actions
#         batch.current['action_distributions'] = action_distributions
#         batch.current['values'] = values
#
#         # return batch
#         return batch


class DynamicModelEvaluator(Evaluator):
    r"""Dynamic model evaluator

    Evaluate the next state given the current state and action.
    """

    def __init__(self, dynamic_model):
        """Initialize the dynamic_model evaluator.

        dynamic_model (DynamicModel): dynamic_model to evaluate.
        """
        # set the dynamic model
        if not isinstance(dynamic_model, DynamicModel):
            raise TypeError("Expecting the given dynamic_model to be an instance of `ActorCritic`, instead got: "
                            "{}".format(type(dynamic_model)))
        self._dynamic_model = dynamic_model

    def _evaluate(self, batch):
        """Evaluate the dynamic_model on the given batch. If None, it will evaluate on the previous batch.

        Args:
            batch (Batch): batch containing the transitions / trajectories.
        """

        # evaluate dynamic_model
        next_states, state_distributions = self._dynamic_model.predict(states=batch['states'], actions=batch['actions'],
                                                                       deterministic=False, to_numpy=False,
                                                                       set_state_data=False)

        # return states and distributions over them
        return [('next_states', next_states), ('state_distributions', state_distributions)]


class ApproximatorEvaluator(Evaluator):
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
        if not isinstance(approximators, list):
            approximators = [approximators]
        for approximator in approximators:
            if not isinstance(approximator, (Approximator, Policy, Value, ActorCritic, DynamicModel, Exploration)):
                raise TypeError("Expecting the approximator to be an instance of `Approximator`, `Policy`, `Value`, "
                                "`ActorCritic`, `DynamicModel`, or `Exploration`. Instead got: "
                                "{}".format(type(approximator)))
        self._approximators = approximators

    def _evaluate(self, batch):
        """Evaluate the various approximators.

        Args:
            batch (Batch): batch containing the transitions / trajectories.
        """

        # sub-evaluation with the current parameter
        outputs = []
        for approximator in self._approximators:
            if isinstance(approximator, (Policy, Exploration)):
                actions, action_distributions = approximator.predict(batch['states'])
                outputs.extend([('actions', actions), ('action_distributions', action_distributions)])
            elif isinstance(approximator, Value):
                values = approximator.evaluate(batch['states'])
                outputs.extend([('values', values)])
            # elif isinstance(approximator, ActorCritic):
            #     actions, action_distributions, values = approximator.evaluate(batch['states'], batch['actions'])
            #     batch.current['actions'] = actions
            #     batch.current['action_distributions'] = action_distributions
            #     batch.current['values'] = values
            elif isinstance(approximator, DynamicModel):
                next_states, state_distributions = approximator.predict(batch['states'], batch['actions'],
                                                                        deterministic=False, to_numpy=False,
                                                                        set_state_data=False)
                outputs.extend([('next_states', next_states), ('state_distributions', state_distributions)])
            else:
                raise TypeError("Expecting the approximator to be an instance of `Policy`, `Value`, `ActorCritic`, or "
                                "`DynamicModel`, instead got: {}".format(type(approximator)))

        return outputs