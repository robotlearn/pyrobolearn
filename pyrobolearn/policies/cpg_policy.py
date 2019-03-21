#!/usr/bin/env python
"""Define the Central Pattern Generator (CPG) Policy.

Define the various CPG policies that can be used.
"""

import numpy as np
import torch

from pyrobolearn.models import CPGNetwork
from pyrobolearn.policies.policy import Policy
from pyrobolearn.states import State, PhaseState
from pyrobolearn.actions import JointAction, Action


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

    def __init__(self, states, actions, rate=1, cpg_network=None, amplitude=np.pi/4, offset=0., init_phase=0.,
                 freq=1., couple_hips=False, parent_coupling=True, child_coupling=True,
                 hip_coupling_weight=None, hip_coupling_bias=0., parent_coupling_weight=None, parent_coupling_bias=0.,
                 child_coupling_weight=None, child_coupling_bias=0., update_amplitudes=True, update_offsets=True,
                 update_init_phases=True, update_frequencies=True, update_weights=True, update_biases=True,
                 amplitude_bounds=np.pi, offset_bounds=np.pi, phase_bounds=np.pi, frequency_bounds=5.,
                 weight_bounds=2., bias_bounds=np.pi, *args, **kwargs):
        """
        Initialize the CPG Network Policy.

        Args:
            states (PhaseState): phase state.
            actions (JointAction): joint action. Normally, it will be JointPositionAction.
            rate (int): number of steps to wait before going to the next step with the policy.
            cpg_network (dict, None): dictionary describing the CPG network. The syntax is the following:
                cpg_network = {<node_id>: {'phi': <phi>, 'offset': <offset>, 'amplitude': <amplitude>, 'freq': <freq>,
                                           'nodes': [{'id': <coupling_node_id>, 'bias': <coupling_bias>,
                                                      'weight': <coupling_weight>}, ...]}
                where <node_id> is the id of the current node, <phi> is the initial phase, <offset> is the initial and
                desired offset, <amplitude> is the amplitude of the signal sent by the CPG node, <freq> is the
                frequency at which operates the node, then we add inside the list associated with the 'nodes' key in
                the dictionary, each node which is coupled to the current node by specifying the id of the coupled
                node, the coupling weight and bias.
                If None, it will create the CPG network automatically by using the other specified arguments.
            amplitude (float, list of float): desired amplitude of each CPG node. If it is a list, it has to match the
                number and order of joints returned by the given `JointAction`.
            offset (float, list of float): desired offset of each CPG node. If it is a list, it has to match the number
                and order of joints returned by the given `JointAction`.
            init_phase (float, list of float): initial phase of each CPG node. If it is a list, it has to match the
                number of joints returned by the given `JointAction`.
            freq (float, list of float): initial desired frequency of each CPG node. If it is a list, it has to match
                the number of joints returned by the given `JointAction`.
            couple_hips (bool): If True it will couple in a bidirectional way the hips together. That is, the phase
                of one hip joint influences the phase of another hip joint.
            parent_coupling (bool): if enabled, then it will couple each node in the leg with its parent. That is,
                the phase of the parent node influences / has an effect on the phase of its child. Assume a leg with
                3 joints [hip, knee, ankle], the phase of the hip will influence the phase of the knee, and the phase
                of the knee will have an effect on the phase of the ankle.
            child_coupling (bool): if enabled, then it will couple each node with its child. That is, the phase of
                the child node influences / has an effect on the phase of its child. Assume a leg with 3 joints [hip,
                knee, ankle], the phase of the ankle influences the phase of the knee, and the phase of the knee has
                an effect on the phase of the hip.
            hip_coupling_weight (float, None): coupling weight between the hips. If None, it will be 1 divided by
                the number of hips.
            hip_coupling_bias (float): hip coupling bias.
            parent_coupling_weight (float, None): coupling weight between the parent and the current node. It is used
                when `parent_coupling` is enabled. If None, it will be 1 divided by the number of joints in the leg.
            parent_coupling_bias (float): parent coupling bias; this bias is used when `parent_coupling` is enabled.
            child_coupling_weight (float, None): coupling weight between the child and the current node. It is used
                when `child_coupling` is enabled. If None, it will be 1 divided by the number of joints in the leg.
            child_coupling_bias (float): child coupling bias; this bias is used when `child_coupling` is enabled.
            update_amplitudes (bool): If True, it will allow to train the desired amplitudes.
            update_offsets (bool): If True, it will allow to train the desired offsets.
            update_frequencies (bool): If True, it will allow to train the desired frequencies.
            update_init_phases (bool): If True, it will allow to optimize the initial phases.
            update_weights (bool): If True, it will allow to optimize the coupling weights.
            update_biases (bool): If True, it will allow to optimize the coupling biases.
            amplitude_bounds (float, tuple of float): bounds / limits to the amplitude parameter (useful when training).
            offset_bounds (float, tuple of float): bounds / limits to the offset parameter (useful when training).
            phase_bounds (float, tuple of float): bounds / limits to the phase parameter (useful when training).
            frequency_bounds (float, tuple of float): bounds / limits to the frequency parameter (useful when training).
            weight_bounds (float, tuple of float): bounds / limits to the weight parameters (useful when training).
            bias_bounds (float, tuple of float): bounds / limits to the bias parameters (useful when training).
            *args (list): other arguments given to the CPG network learning model.
            **kwargs (dict): other key + value arguments given to the CPG network learning model.
        """
        super(CPGPolicy, self).__init__(states, actions, rate=rate, *args, **kwargs)

        # check actions
        if not isinstance(actions, JointAction):
            raise TypeError("Expecting the actions to be an instance of JointAction, instead got: "
                            "{}".format(type(actions)))

        # check states
        if not isinstance(states, PhaseState):
            raise TypeError("Expecting the states to be an instance of PhaseState, instead got: "
                            "{}".format(type(states)))

        # get useful information from the state/action
        timesteps = states.num_steps
        robot = actions.robot
        joints = set(actions.joints)

        # create CPG network based on the robot kinematic structures if not provided
        if cpg_network is None:
            # get specified legs
            legs = []
            for robot_leg in robot.legs:
                leg = []
                for joint in robot_leg:
                    if joint in joints:
                        leg.append(joint)
                legs.append(leg)

            num_legs = len(legs)

            # variables for coupling the nodes
            if couple_hips:
                if hip_coupling_weight is None:
                    hip_coupling_weight = 1. / num_legs

            init_parent_coupling_weight = parent_coupling_weight
            init_child_coupling_weight = child_coupling_weight

            # create the CPG network based on the robot kinematic structures
            cpg_network = {}
            for leg_idx, leg in enumerate(legs):
                # compute parent/child leg coupling weight
                if len(leg) != 0:
                    if parent_coupling and init_parent_coupling_weight is None:
                        parent_coupling_weight = 1. / len(leg)
                    if child_coupling and init_child_coupling_weight is None:
                        child_coupling_weight = 1. / len(leg)

                for idx, joint in enumerate(leg):
                    # proper node parameters
                    node = {'phi': init_phase, 'offset': offset, 'amplitude': amplitude, 'freq': freq}

                    # coupling parameters

                    # if first upper joint in the leg, connect it with the other upper joints (in the other legs)
                    if couple_hips and idx == 0:
                        for l in legs:
                            if len(l) > 0 and joint != l[0]:  # i.e. not the same node
                                coupling_params = {'id': l[0], 'weight': hip_coupling_weight, 'bias': hip_coupling_bias}
                                node.setdefault('nodes', []).append(coupling_params)

                    # add coupling to current joint (except the last one) with the next joint in the leg
                    if parent_coupling and idx < len(leg)-1:
                        # add next node
                        coupling_params = {'id': leg[idx + 1], 'weight': parent_coupling_weight,
                                           'bias': parent_coupling_bias}
                        node.setdefault('nodes', []).append(coupling_params)

                    # add coupling to current joint (except the first one) with the previous joint in the leg
                    if child_coupling and idx > 0:
                        # add next node
                        coupling_params = {'id': leg[idx - 1], 'weight': child_coupling_weight,
                                           'bias': child_coupling_bias}
                        node.setdefault('nodes', []).append(coupling_params)

                    # add node in the CPG network dictionary
                    cpg_network[joint] = node
        else:
            # Quick check
            if isinstance(joints, int):
                joints = [joints]
            if len(cpg_network) != len(joints):
                cpg_network_ids = [idx for idx in cpg_network]
                raise ValueError("The number of joints doesn't match up the number of CPG id in the CPG network. "
                                 "The joints specified in the joint actions are {}, and the joint ids in the CPG "
                                 "network are {}".format(joints, cpg_network_ids))

        self.cpg_network = cpg_network

        # create learning model
        self.model = CPGNetwork(nodes=cpg_network, timesteps=timesteps, update_amplitudes=update_amplitudes,
                                update_offsets=update_offsets, update_init_phases=update_init_phases,
                                update_frequencies=update_frequencies, update_weights=update_weights,
                                update_biases=update_biases, amplitude_bounds=amplitude_bounds,
                                offset_bounds=offset_bounds, phase_bounds=phase_bounds,
                                frequency_bounds=frequency_bounds, weight_bounds=weight_bounds,
                                bias_bounds=bias_bounds, *args, **kwargs)

    def _size(self, x):
        size = 0
        if isinstance(x, (State, Action)):
            if x.is_discrete():
                size = x.space[0].n
            else:
                size = x.total_size()
        elif isinstance(x, np.ndarray):
            size = x.size
        elif isinstance(x, torch.Tensor):
            size = x.numel()
        elif isinstance(x, int):
            size = x
        return size

    def act(self, state=None, deterministic=True, to_numpy=True, return_logits=False):
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

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a string that describes the CPG policy."""
        description = self.__class__.__name__ + '(' + str(self.model) + ')'
        return description
