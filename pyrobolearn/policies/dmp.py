#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Dynamic Movement Primitive (DMP) Policy.

Define the various DMP policies that can be used.
"""

import numpy as np
import torch

from pyrobolearn.models import DMP, DiscreteDMP, RhythmicDMP, BioDiscreteDMP
from pyrobolearn.policies.policy import Policy
from pyrobolearn.states import State
from pyrobolearn.actions import Action, JointPositionAction, JointVelocityAction, JointAccelerationAction, \
    JointPositionAndVelocityAction

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DMPPolicy(Policy):
    r"""Dynamic Movement Primitive (DMP) policy
    """

    def __init__(self, state, action, model, rate=1, preprocessors=None, postprocessors=None, *args, **kwargs):
        """
        Initialize the DMP policy.

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
            model (DMP): DMP model
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
              stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
              executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
            *args (list): list of arguments (this is not used in this class).
            **kwargs (dict): dictionary of arguments (this is not used in this class).
        """
        if not isinstance(model, DMP):
            raise TypeError("Expecting model to be an instance of `DMP`, but got instead: {}".format(type(model)))
        super(DMPPolicy, self).__init__(state=state, action=action, model=model, rate=rate,
                                        preprocessors=preprocessors, postprocessors=postprocessors, *args, **kwargs)

        # check actions
        self.is_joint_position_action = JointPositionAction in action or JointPositionAndVelocityAction in action
        self.is_joint_velocity_action = JointVelocityAction in action or JointPositionAndVelocityAction in action
        self.is_joint_acceleration_action = JointAccelerationAction in action
        if not (self.is_joint_position_action or self.is_joint_velocity_action or self.is_joint_acceleration_action):
            raise ValueError("The actions do not have a joint position, velocity, or acceleration action.")

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
        Plot the rollouts using the DMPs.

        Args:
            nrows (int): number of rows in the subplot.
            ncols (int): number of columns in the subplot.
            suptitle (str): main title for the subplots.
            titles (str, list[str]): title for each subplot.
            show (bool): if True, it will show and block the plot.
        """
        self.model.plot_rollout(nrows=nrows, ncols=ncols, suptitle=suptitle, titles=titles, show=show)


class DiscreteDMPPolicy(DMPPolicy):
    r"""Discrete DMP Policy

    See Also: see documentation in `pyrobolearn.models.dmp.discrete_dmp.py`

    References:
        - [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
    """

    def __init__(self, action, state=None, num_basis=20, dt=0.01, y0=0, goal=1, forcing_terms=None,
                 stiffness=None, damping=None, rate=1):
        """
        Initialize the discrete DMP policy.

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
            dt (float): step integration for Euler's method
            y0 (float, np.array): initial position(s)
            goal (float, np.array): goal(s)
            forcing_terms (list, ForcingTerm): the forcing terms (which can have different basis functions)
            stiffness (float): stiffness coefficient
            damping (float): damping coefficient
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
              stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
              executing the model.
        """
        if not isinstance(action, Action):
            raise TypeError("Expecting actions to be an instance of the 'Action' class.")
        model = DiscreteDMP(num_dmps=self._size(action), num_basis=num_basis, dt=dt, y0=y0, goal=goal,
                            forcing_terms=forcing_terms, stiffness=stiffness, damping=damping)
        super(DiscreteDMPPolicy, self).__init__(state, action, model, rate=rate)


class RhythmicDMPPolicy(DMPPolicy):
    r"""Rhythmic DMP Policy

    See Also: see documentation in `pyrobolearn.models.dmp.rhythmic_dmp.py`

    References:
        - [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
    """

    def __init__(self, action, state=None, num_basis=20, dt=0.01, y0=0, goal=1, forcing_terms=None,
                 stiffness=None, damping=None, rate=1):
        """
        Initialize the Rhythmic DMP policy.

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
            dt (float): step integration for Euler's method
            y0 (float, np.array): initial position(s)
            goal (float, np.array): goal(s)
            forcing_terms (list, ForcingTerm): the forcing terms (which can have different basis functions)
            stiffness (float): stiffness coefficient
            damping (float): damping coefficient
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
              stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
              executing the model.
        """
        model = RhythmicDMP(num_dmps=self._size(action), num_basis=num_basis, dt=dt, y0=y0, goal=goal,
                            forcing_terms=forcing_terms, stiffness=stiffness, damping=damping)
        super(RhythmicDMPPolicy, self).__init__(state, action, model, rate=rate)


class BioDiscreteDMPPolicy(DMPPolicy):
    r"""Bio Discrete DMP Policy

    See Also: see documentation in `pyrobolearn.models.dmp.biodiscrete_dmp.py`

    References:
        - [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
        - [2] "Biologically-inspired Dynamical Systems for Movement Generation: Automatic Real-time Goal Adaptation
          and Obstacle Avoidance", Hoffmann et al., 2009
        - [3] "Learning and Generalization of Motor Skills by Learning from Demonstration", Pastor et al., 2009
    """

    def __init__(self, action, state=None, num_basis=20, dt=0.01, y0=0, goal=1, forcing_terms=None,
                 stiffness=None, damping=None, rate=1):
        """
        Initialize the biologically-inspired DMP policy.

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
            dt (float): step integration for Euler's method
            y0 (float, np.array): initial position(s)
            goal (float, np.array): goal(s)
            forcing_terms (list, ForcingTerm): the forcing terms (which can have different basis functions)
            stiffness (float): stiffness coefficient
            damping (float): damping coefficient
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
              stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
              executing the model.
        """
        model = BioDiscreteDMP(num_dmps=self._size(action), num_basis=num_basis, dt=dt, y0=y0, goal=goal,
                               forcing_terms=forcing_terms, stiffness=stiffness, damping=damping)
        super(BioDiscreteDMPPolicy, self).__init__(state, action, model, rate=rate)
