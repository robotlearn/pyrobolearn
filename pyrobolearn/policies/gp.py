#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Gaussian Process Regression (GPR) Policy.

Define the GPR policy that can be used.
"""

import numpy as np
import torch

from pyrobolearn.approximators.gp import GPRApproximator
from pyrobolearn.policies.policy import Policy
from pyrobolearn.states import State
from pyrobolearn.actions import Action


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GPRPolicy(Policy):
    r"""Gaussian Process Regression (GPR) policy
    """

    def __init__(self, state, action, mean=None, kernel=None, model=None, likelihood=None, rate=1, preprocessors=None,
                 postprocessors=None, *args, **kwargs):
        """
        Initialize the GPR policy.

        Args:
            action (Action): At each step, by calling `policy.act(state)`, the `action` is computed by the policy,
              and can be given to the environment. As with the `state`, the type and size/shape of each inner
              action can be inferred and could be used to automatically build a policy. The `action` connects the
              policy with a controllable object (such as a robot) in the environment.
            state (State): By giving the `state` to the policy, it can automatically infer the type and size/shape
              of each inner state, and thus can be used to automatically build a policy. At each step, the `state`
              is filled by the environment, and read by the policy. The `state` connects the policy with one or
              several objects (including robots) in the environment. Note that some policies don't use any state
              information.
            mean (None, gpytorch.means.Mean): mean prior. If None, it will be set to `gpytorch.means.ConstantMean()`.
            kernel (None, gpytorch.kernels.Kernel): kernel prior. If None it will be set to
              `gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.WhiteNoiseKernel())`
            model (None, gpytorch.module.Module): the prior GP model. If None, it will create `ExactGPModel()`, a GP
              model using the provided mean, kernel, and likelihood.
            likelihood (None, gpytorch.likelihoods.Likelihood): the likelihood pdf. If None, it will use the
              `gpytorch.likelihoods.GaussianLikelihood()`.
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
              stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
              executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
            *args (list): list of arguments (this is not used in this class).
            **kwargs (dict): dictionary of arguments (this is not used in this class).
        """
        model = GPRApproximator(inputs=state, outputs=action, mean=mean, kernel=kernel, model=model,
                                likelihood=likelihood, preprocessors=preprocessors,
                                postprocessors=postprocessors)
        super(GPRPolicy, self).__init__(state, action, model, rate=rate, *args, **kwargs)

    def inner_predict(self, state, deterministic=True, to_numpy=False, return_logits=True, set_output_data=False):
        """Inner prediction step.

        Args:
            state ((list of) torch.Tensor, (list of) np.array): state data.
            deterministic (bool): if True, it will predict in a deterministic way. Setting it to False, only works
              with stochastic models.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
            return_logits (bool): If True, in the case of discrete outputs, it will return the logits.
            set_output_data (bool): If True, it will set the predicted output data to the outputs given to the
              approximator.

        Returns:
            (list of) torch.Tensor, (list of) np.array: predicted action data.
        """
        if isinstance(state, (np.ndarray, list, tuple)):
            state = state[0]
        y = self.model.condition(x_in=state, idx_out=None)  # TODO
        return y

    # def act(self, state=None, deterministic=True, to_numpy=True, return_logits=False, apply_action=True):
    #     # return self.model.predict(state, to_numpy=to_numpy)
    #     if (self.cnt % self.rate) == 0:
    #         # print("Policy state value: {}".format(state.data[0][0]))
    #         self.y, self.dy, self.ddy = self.model.step(state.data[0][0])
    #     self.cnt += 1
    #     # y, dy, ddy = self.model.step()
    #     # return np.array([y, dy, ddy])
    #     if isinstance(self.actions, JointPositionAction):
    #         # print("DMP action: {}".format(self.y))
    #         self.actions.data = self.y
    #     elif isinstance(self.actions, JointVelocityAction):
    #         self.actions.data = self.dy
    #     elif isinstance(self.actions, JointAccelerationAction):
    #         self.actions.data = self.ddy
    #     return self.actions

    # def sample(self, state):
    #     pass

    def rollout(self):
        """Perform a rollout with the movement primitive."""
        return self.model.rollout()

    def imitate(self, data):  # TODO: improve this
        if len(data) > 0:
            raise NotImplementedError
        else:
            print("Nothing to imitate.")

    def plot_rollout(self, nrows=1, ncols=1, suptitle=None, titles=None, show=True):
        """
        Plot the rollouts using the DMPs.

        Args:
            nrows (int): number of rows in the subplot.
            ncols (int): number of columns in the subplot.
            suptitle (str): main title for the subplots.
            titles (str, list[str]): title for each subplot.
            show (bool): if True, it will show and block the plot.
        """
        self.model.plot_rollout(nrows=nrows, ncols=ncols, suptitle=suptitle, titles=titles, show=show)
