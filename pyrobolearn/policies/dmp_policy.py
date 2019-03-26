#!/usr/bin/env python
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
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DMPPolicy(Policy):
    r"""Dynamic Movement Primitive (DMP) policy
    """

    def __init__(self, states, actions, model, rate=1, *args, **kwargs):
        if not isinstance(model, DMP):
            raise TypeError("Expecting model to be an instance of DMP")
        super(DMPPolicy, self).__init__(states, actions, model, rate=rate, *args, **kwargs)

        # check actions
        self.is_joint_position_action = JointPositionAction in actions or JointPositionAndVelocityAction in actions
        self.is_joint_velocity_action = JointVelocityAction in actions or JointPositionAndVelocityAction in actions
        self.is_joint_acceleration_action = JointAccelerationAction in actions
        if not (self.is_joint_position_action or self.is_joint_velocity_action or self.is_joint_acceleration_action):
            raise ValueError("The actions do not have a joint position, velocity, or acceleration action.")

    def _predict(self, state, to_numpy=False, return_logits=True, set_output_data=False):
        """Inner prediction step."""
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
            y = data
            # if len(y.shape) == 1:
            #     y = y.reshape(1, -1)
            # if len(dy.shape) == 1:
            #     dy = dy.reshape(1, -1)
            # if len(ddy.shape) == 1:
            #     ddy = ddy.reshape(1, -1)
            self.model.imitate(y, plot=False)  # dy, ddy, plot=True)  # dy, ddy)
        else:
            print("Nothing to imitate.")


class DiscreteDMPPolicy(DMPPolicy):
    r"""Discrete DMP Policy
    """

    def __init__(self, actions, states=None, num_basis=20, dt=0.01, y0=0, goal=1, forcing_terms=None,
                 stiffness=None, damping=None, rate=1):
        if not isinstance(actions, Action):
            raise TypeError("Expecting actions to be an instance of the 'Action' class.")
        model = DiscreteDMP(num_dmps=self._size(actions), num_basis=num_basis, dt=dt, y0=y0, goal=goal,
                            forcing_terms=forcing_terms, stiffness=stiffness, damping=damping)
        super(DiscreteDMPPolicy, self).__init__(states, actions, model, rate=rate)


class RhythmicDMPPolicy(DMPPolicy):
    r"""Rhythmic DMP Policy
    """

    def __init__(self, actions, states=None, num_basis=20, dt=0.01, y0=0, goal=1, forcing_terms=None,
                 stiffness=None, damping=None, rate=1):
        model = RhythmicDMP(num_dmps=self._size(actions), num_basis=num_basis, dt=dt, y0=y0, goal=goal,
                            forcing_terms=forcing_terms, stiffness=stiffness, damping=damping)
        super(RhythmicDMPPolicy, self).__init__(states, actions, model, rate=rate)


class BioDiscreteDMPPolicy(DMPPolicy):
    r"""Bio Discrete DMP Policy
    """

    def __init__(self, actions, states=None, num_basis=20, dt=0.01, y0=0, goal=1, forcing_terms=None,
                 stiffness=None, damping=None, rate=1):
        model = BioDiscreteDMP(num_dmps=self._size(actions), num_basis=num_basis, dt=dt, y0=y0, goal=goal,
                               forcing_terms=forcing_terms, stiffness=stiffness, damping=damping)
        super(BioDiscreteDMPPolicy, self).__init__(states, actions, model, rate=rate)
