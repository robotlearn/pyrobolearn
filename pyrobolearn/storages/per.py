#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides the prioritized experience replay (PER) storage.

The PER works by prioritizing transitions based on the magnitude of their TD error. In order to overcome over-fitting
by sampling the same transitions, a stochastic sampling method is used based on a kind of softmax function on TD
errors. In order to correct the bias induced by this sampling method, importance sampling weights (which are
normalized for stability reasons) are used.

In summary, PER can be seen as a stochastic prioritization ER which uses importance sampling.

References:
    - [1] "Prioritized Experience Replay", Schaul, 2015
"""

from pyrobolearn.storages.storage import PriorityQueueStorage

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PrioritizedExperienceReplay(PriorityQueueStorage):
    r"""Prioritized Experience Replay storage

    The PER works by prioritizing transitions based on the magnitude of their TD error. In order to overcome
    over-fitting by sampling the same transitions, a stochastic sampling method is used based on a kind of softmax
    function on TD errors. In order to correct the bias induced by this sampling method, importance sampling weights
    (which are normalized for stability reasons) are used.

    In summary, PER can be seen as a stochastic prioritization ER which uses importance sampling.

    There are 2 stochastic prioritization schemes used in [1]:

    - proportional prioritization: :math:`p_i = |\delta_i| + \epsilon`, where :math:`\epsilon` is a small positive
    constant to avoid the transition to have a probability of 0.

    - rank-based prioritization: math:`p_i = \frac{1}{rank(i)}`, where rank(i) is the rank of transition i (that is
    they are i other keys in the priority queue that are smaller than the current key i) when the replay memory is
    sorted according to :math:`|\delta_i|`


    Pseudo-algo:
    -----------

    Pseudo-algorithm (taken from [1] and reproduce here for completeness)::
    1. Input: minibatch k, step-size \eta, replay period K and size N, exponents a and b, budget T
    2. Initialize replay memory H = {}, \Delta = 0, p_1 = 1
    3. Observe s_0 and choose a_1 \sim \pi_\theta(s_0)
    4. for t = 0 to T-1 do
    5.     choose action a_t \sim \pi_\theta(s_t)
    6.     Observe s_{t+1}, r_t, \gamma_t
    7.     Store transition (s_t, a_t, r_t, s_{t+1}, \gamma_t) in H with maximal priority p_t = max_{i<t} p_i
    8.     if (t % K) = 0 then
    9.         for j = 1 to k do
    10.            Sample transition j \sim P(j) = \frac{ p_j^a }{ \sum_i p_i^a }
    11.            Compute importance-sampling weight w_j = \frac{ (N P(j))^{-b} }{ \max_i w_i }
    12.            Compute TD-error \delta_j = r_j + \gamma_j Q_{target}(s_j, argmax_a Q(s_j, a)) - Q(s_{j-1}, a_{j-1})
    13.            Update transition priority p_j \leftarrow |\delta_j|
    14.            Accumulate weight-change \Delta \leftarrow \Delta + w_j \delta_j \nabla_\theta Q(s_{j-1},a_{j-1})
    15.        end for
    16.        Update weights \theta \leftarrow \theta + \eta \Delta, reset \Delta = 0
    17.        From time to time copy weights into target network \theta_{target} \leftarrow \theta
    18.     end if
    19. end for


    References:
        - [1] "Prioritized Experience Replay", Schaul, 2015
    """

    def __init__(self):
        super(PrioritizedExperienceReplay, self).__init__()


# alias
PER = PrioritizedExperienceReplay
