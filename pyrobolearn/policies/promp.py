#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Probabilistic Movement Primitive (ProMP) Policy.

Define the various ProMP policies that can be used.
"""

import numpy as np
import torch

from pyrobolearn.models import ProMP, DiscreteProMP, RhythmicProMP
from pyrobolearn.policies.policy import Policy
from pyrobolearn.states import State
from pyrobolearn.actions import Action, JointPositionAction, JointVelocityAction, JointPositionAndVelocityAction

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ProMPPolicy(Policy):
    r"""Probabilistic Movement Primitive (ProMP) policy
    """

    def __init__(self, state, action, model, rate=1, preprocessors=None, postprocessors=None, *args, **kwargs):
        """
        Initialize the ProMP policy.

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
            model (ProMP): ProMP model
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
              stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
              executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
            *args (list): list of arguments (this is not used in this class).
            **kwargs (dict): dictionary of arguments (this is not used in this class).
        """
        if not isinstance(model, ProMP):
            raise TypeError("Expecting model to be an instance of `ProMP`, but got instead: {}".format(type(model)))
        super(ProMPPolicy, self).__init__(state=state, action=action, model=model, rate=rate,
                                          preprocessors=preprocessors, postprocessors=postprocessors, *args, **kwargs)

        # check actions
        self.is_joint_position_action = JointPositionAction in action or JointPositionAndVelocityAction in action
        self.is_joint_velocity_action = JointVelocityAction in action or JointPositionAndVelocityAction in action
        if not (self.is_joint_position_action or self.is_joint_velocity_action):
            raise ValueError("The actions do not have a joint position or velocity action.")

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
        y, dy, ddy = self.model.step(state)
        if self.is_joint_position_action:
            if self.is_joint_velocity_action:
                return np.concatenate((y, dy))
            return y
        elif self.is_joint_velocity_action:
            return dy
        else:  # self.is_joint_acceleration_action
            return ddy

    # def act(self, state=None, deterministic=True, to_numpy=True, return_logits=False, apply_action=True):
    #     # return self.model.predict(state, to_numpy=to_numpy)
    #     if (self.cnt % self.rate) == 0:
    #         # print("Policy state value: {}".format(state.data[0][0]))
    #         self.y, self.dy, self.ddy = self.model.step(state.data[0][0])
    #     self.cnt += 1
    #     # y, dy, ddy = self.model.step()
    #     # return np.array([y, dy, ddy])
    #     if isinstance(self.actions, JointPositionAction):
    #         # print("ProMP action: {}".format(self.y))
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
            # print("Imitating with :", data.shape)
            # y, dy, ddy = data
            # y = data
            # if len(y.shape) == 1:
            #     y = y.reshape(1, -1)
            # if len(dy.shape) == 1:
            #     dy = dy.reshape(1, -1)
            # if len(ddy.shape) == 1:
            #     ddy = ddy.reshape(1, -1)
            if self.is_joint_position_action:
                if self.is_joint_velocity_action:
                    self.model.imitate(y, dy, plot=False)
                else:
                    y = data
                    self.model.imitate(y, plot=False)  # dy, ddy, plot=True)  # dy, ddy)
            else:
                raise NotImplementedError
        else:
            print("Nothing to imitate.")

    def plot_rollout(self, nrows=1, ncols=1, suptitle=None, titles=None, show=True):
        """
        Plot the rollouts using the ProMPs.

        Args:
            nrows (int): number of rows in the subplot.
            ncols (int): number of columns in the subplot.
            suptitle (str): main title for the subplots.
            titles (str, list[str]): title for each subplot.
            show (bool): if True, it will show and block the plot.
        """
        self.model.plot_rollout(nrows=nrows, ncols=ncols, suptitle=suptitle, titles=titles, show=show)


class DiscreteProMPPolicy(ProMPPolicy):
    r"""Discrete ProMP Policy

    See Also: see documentation in `pyrobolearn.models.promp.discrete_promp.py`

    References:
        - [1] "Probabilistic Movement Primitives", Paraschos et al., 2013
        - [2] "Using Probabilistic Movement Primitives in Robotics", Paraschos et al., 2018
    """

    def __init__(self, action, state=None, num_basis=20, weights=None, canonical_system=None, noise_covariance=1.,
                 basis_width=None, rate=1):
        """
        Initialize the discrete ProMP policy.

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
            num_basis (int): number of basis functions

            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
              stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
              executing the model.
        """
        if not isinstance(action, Action):
            raise TypeError("Expecting actions to be an instance of the 'Action' class.")
        model = DiscreteProMP(num_dofs=self._size(action), num_basis=num_basis, weights=weights,
                              canonical_system=canonical_system, noise_covariance=noise_covariance,
                              basis_width=basis_width)
        super(DiscreteProMPPolicy, self).__init__(state, action, model, rate=rate)


class RhythmicProMPPolicy(ProMPPolicy):
    r"""Rhythmic ProMP Policy

    See Also: see documentation in `pyrobolearn.models.promp.rhythmic_promp.py`

    References:
        - [1] "Probabilistic Movement Primitives", Paraschos et al., 2013
        - [2] "Using Probabilistic Movement Primitives in Robotics", Paraschos et al., 2018
    """

    def __init__(self, action, state=None, num_basis=20, weights=None, canonical_system=None, noise_covariance=1.,
                 basis_width=None, rate=1):
        """
        Initialize the Rhythmic ProMP policy.

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
            num_basis (int): number of basis functions

            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
              stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
              executing the model.
        """
        model = RhythmicProMP(num_dofs=self._size(action), num_basis=num_basis, weights=weights,
                              canonical_system=canonical_system, noise_covariance=noise_covariance,
                              basis_width=basis_width)
        super(RhythmicProMPPolicy, self).__init__(state, action, model, rate=rate)
