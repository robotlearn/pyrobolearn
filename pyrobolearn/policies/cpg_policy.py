#!/usr/bin/env python
"""Define the Central Pattern Generator (CPG) Policy.

Define the various CPG policies that can be used.
"""

import numpy as np
import torch

from pyrobolearn.models import CPGNetwork
from policy import Policy
from pyrobolearn.states import State
from pyrobolearn.actions import Action, JointPositionAction


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CPGPolicy(Policy):
    r"""Central Pattern Generator (CPG) Network policy
    """

    def __init__(self, states, actions, timesteps=100, rate=1, *args, **kwargs):
        super(CPGPolicy, self).__init__(states, actions, rate=rate, *args, **kwargs)

        # check actions
        if not isinstance(actions, JointPositionAction):
            raise TypeError("Expecting the actions to be an instance of JointPositionAction, instead got: "
                            "{}".format(type(actions)))

        # create CPG network based on the robot kinematic structures

        # get specified legs
        robot = actions.robot
        joints = set(actions.joints)
        legs = []
        for robot_leg in robot.legs:
            leg = []
            for joint in robot_leg:
                if joint in joints:
                    leg.append(joint)
            legs.append(leg)

        num_legs = len(legs)

        # define few variables to initialize the CPG nodes
        # variables for the node
        init_phi = 0.
        offset = 0.
        amplitude = 1.
        freq = 1.
        # variables for coupling the nodes
        weight_legs = 1. / len(legs)
        if len(legs) > 0 and len(legs[0]) > 0:
            weight_leg = 1. / len(legs[0])
        else:
            weight_leg = 0.
        bias = 0.

        # create the CPG network based on the robot kinematic structures
        nodes = {}
        for leg_idx, leg in enumerate(legs):
            for idx, joint in enumerate(leg):
                # proper node parameters
                node = {'phi': init_phi, 'offset': offset, 'amplitude': amplitude, 'freq': freq}

                # coupling parameters

                # if first upper joint in the leg, connect it with the other upper joints (in the other legs)
                if idx == 0:
                    for l in legs:
                        if len(l) > 0 and joint != l[0]:  # i.e. not the same node
                            coupling_params = {'id': l[0], 'weight': weight_legs, 'bias': bias}
                            node.setdefault('nodes', []).append(coupling_params)

                # add coupling to current joint (except the last one) with the next joint in the leg
                if idx < len(leg)-1:
                    # add next node
                    coupling_params = {'id': leg[idx + 1], 'weight': weight_leg, 'bias': bias}
                    node.setdefault('nodes', []).append(coupling_params)

                # add coupling to current joint (except the first one) with the previous joint in the leg
                if idx > 0:
                    # add next node
                    coupling_params = {'id': leg[idx - 1], 'weight': weight_leg, 'bias': bias}
                    node.setdefault('nodes', []).append(coupling_params)

                # add node in the CPG network dictionary
                nodes[joint] = node

        # create learning model
        self.model = CPGNetwork(nodes=nodes, timesteps=timesteps)

    def _size(self, x):
        size = 0
        if isinstance(x, (State, Action)):
            if x.isDiscrete():
                size = x.space[0].n
            else:
                size = x.totalSize()
        elif isinstance(x, np.ndarray):
            size = x.size
        elif isinstance(x, torch.Tensor):
            size = x.numel()
        elif isinstance(x, int):
            size = x
        return size

    def act(self, state=None, deterministic=True, to_numpy=True):
        if (self.cnt % self.rate) == 0:
            self.last_action = self.model.step()
        self.cnt += 1
        # angles = self.model.step()
        # self.actions.data = angles
        self.actions.data = self.last_action
        return self.actions

    def sample(self, state):
        pass

    def phase_resetting(self):
        self.model.reset()
        pass
