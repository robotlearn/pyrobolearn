#!/usr/bin/env python
"""Provides the hindsight experience replay (HER) storage.

References:
    [1] "Hindsight Experience Replay", Andrychowicz et al., 2017
"""

from pyrobolearn.storages.er import ExperienceReplay

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class HindsightExperienceReplay(ExperienceReplay):
    r"""Hindsight Experience replay storage

    One of the main challenges in RL is to shape the reward function such that the agent can successfully learned to
    perform the specified task. This often requires expert knowledge to engineer this reward function.
    To address this, the authors from [1] proposes to use a hindsight experience replay, which enables learning from
    sparse and binary rewards, and can be combined with any off-policy RL algorithms. This notably improves the
    sample efficiency.

    In this setting, one or several goals have to be defined. They are concatenated with the state and feed to the
    policy and value approximators. Additionally, they are included in the transition tuple sampled from the
    experience replay storage.


    Pseudo-algo
    -----------

    Pseudo-algorithm (taken from [1] and reproduce here for completeness)::
    1. Given:
        - an off-policy RL algorithm A (e.g. DQN, DDPG, NAF, SDQN)
        - a strategy S for sampling goals for replay (e.g. S(s_0, ..., s_T) = m(s_T))
        - a reward function r : S x A x G \rightarrow \mathbb{R} (e.g. r(s, a, g) = -[f_g(s) = 0])
    2. Initialize A (e.g. initialize neural networks)
    3. Initialize replay buffer R
    4. for episode = 1 to M do
    5.     Sample a goal g and an initial state s0.
    6.     for t = 0 to T - 1 do
    7.         Sample an action at using the behavioral policy from A: a_t \leftarrow \pi_b([s_t,g])
    8.         Execute the action a_t and observe a new state s_{t+1}
    9.     end for
    10.    for t = 0 to T - 1 do
    11.        r_t := r(s_t, a_t, g)
    12.        Store the transition ([s_t,g], a_t, r_t, [s_{t+1},g]) in R (standard experience replay)
    13.        Sample a set of additional goals for replay G := S(current episode)
    14.        for g' \in G do
    15.            r' := r(s_t, a_t, g')
    16.            Store the transition ([s_t,g'], a_t, r', [s_{t+1},g']) in R (HER)
    17.        end for
    18.     end for
    19.     for t = 1 to N do
    20.         Sample a minibatch B from the replay buffer R
    21.         Perform one step of optimization using A and minibatch B
    22.     end for
    23. end for


    References:
        [1] "Hindsight Experience Replay", Andrychowicz et al., 2017
    """
    pass


# alias
HER = HindsightExperienceReplay
