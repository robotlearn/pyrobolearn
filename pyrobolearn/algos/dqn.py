#!/usr/bin/env python
"""Provide the DQN algorithm.

The DQN is a value-based, off-policy, model-free RL algorithm with exploration in the action space.
"""

import copy
import torch

# from pyrobolearn.envs import Env
from pyrobolearn.policies import PolicyFromQValue
# from pyrobolearn.tasks import RLTask
from pyrobolearn.algos.rl_algo import GradientRLAlgo, Explorer, Evaluator, Updater

from pyrobolearn.values import ParametrizedQValueOutput
from pyrobolearn.exploration import EpsilonGreedyActionExploration

from pyrobolearn.storages import ExperienceReplay
from pyrobolearn.returns import TDQLearningReturn
from pyrobolearn.losses import MSBELoss, HuberLoss
from pyrobolearn.optimizers import Adam

from pyrobolearn.parameters.updater import CopyParameter


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse", "PyTorch (Adam Paszke)"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DQN(GradientRLAlgo):
    r"""Deep Q-Network

    Type: model-free, value-based, off-policy RL algorithm for discrete action spaces with exploration in the
    action space.


    Background
    ----------

    Deep Q-Networks (DQNs) work by approximating the Q-value function using a deep neural network [1,2]. The optimal
    policy is then inferred from it by returning the discrete action that maximizes the Q-value function, i.e.
    :math:`\pi^*(s_t) = argmax_{a_t} Q(s_t,a_t)`.The DQN is trained by minimizing the mean-squared Bellman error loss,
    which minimizes the TD-error, and thus enforce the validity of Bellman's equations.

    The loss is given by:

    .. math:: \mathcal{L}(\phi) = \mathbb{E}_{s,a}[(y - Q_{\phi}(s,a))^2]

    where the targets are given by :math:`y = r(s_t, a_t) + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a')`. The
    Q-learning TD(0) error is given by :math:`\delta = y - Q_{\phi}(s,a)` and thus,
     :math:`\delta = (r(s_t, a_t) + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a') - Q_{\phi}(s,a))`

    There are three important notes:
    1. In order to select the action that maximizes the Q-values, they have to be discrete. The Q-value function
    approximator thus accepts as input the state and outputs for each action its corresponding Q-value.
    2. The data distribution that we have is dependent of the policy that we are optimizing. Additionally, the
    generated samples are highly correlated in time. In order to address these issues and have i.i.d data samples, an
    experience replay (ER) memory is used from which transition tuples :math:`(s_t, a_t, s_{t+1}, r_t, d, \gamma)` are
    sampled (uniformly in general). Its necessity and utility has been shown in [1,2,3].
    3. Using the same Q-value function approximator to evaluate the targets lead to instable behaviors [1,2]. In order
    to improve stability during the training, an old Q-value function is thus used when computing the targets and
    updated every once in a while (at a lesser frequency than the current Q-value function approximator).

    Exploration is usually performed in the action space by using :math:`\epsilon`-greedy exploration or a Boltzmann
    policy.


    Pseudo-algo
    -----------

    Pseudo-algorithm using DQN, an experience replay and :math:`\epsilon`-greedy exploration (mainly taken from [1],
    modified a bit, and reproduced here for completeness)::

    1. Initialize replay memory :math:`D` to capacity N
    2. Initialize action-value function Q with random weights :math:`\phi_0`
    3. for episode = 1 to M do
    4.     for t = 1 to T do
    5.         With probability :math:`\epsilon` select a random action :math:`a_t` otherwise select
               :math:`a_t = argmax_a Q_{\phi_{target}}(s_t, a)`
    6.         Execute action :math:`a_t`in environment and observe reward :math:`r_t`, next state :math:`s_{t+1}`, and
               the binary signal :math:`d` if the task is done (d=1) or not (d=0).
    7.         Store transition :math:`(s_t, a_t, r_t, s_{t+1}, d)` in D
    8.     for batch = 1 to K do
    9.         Sample random minibatch of transitions :math:`(s_t, a_t, r_t, s_{t+1}, d)` from D
    10.        Compute target: y= r_t + \gamma (1 - d) \max_{a'} Q_{\phi_{target}}(s_{t+1}, a')
    11.        Minimize MSBE loss :math:`(y - Q_{\phi}(s_t,a_t))^2` with respect to the parameters :math:`\phi`, by
               performing a descent gradient step (for instance)
    12.        Update :math:`\phi_{target}` every C steps (:math:`\phi_{target} = \phi`) or using polyak averaging
               (:math:`\phi_{target} = \rho \phi_{target} + (1 - \rho) \phi`)
    13.    end for
    14. end for


    References:
        [1] "Playing Atari with Deep Reinforcement Learning", Mnih et al., 2013
        [2] "Human-level Control through Deep Reinforcement Learning", Mnih et al., 2015
        [3] "Reinforcement Learning for robots using neural networks", Lin, 1993
        [4] "Reinforcement Learning (DQN) Tutorial" (in PyTorch):
            https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        [5] "A (Long) Peek into Reinforcement Learning":
            https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html
        [6] "Implementing Deep Reinforcement Learning Models with Tensorflow + OpenAI Gym":
            https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html
    """

    def __init__(self, task, approximator, gamma=0.99, lr=5e-4, capacity=10000, num_workers=1):
        """
        Initialize the DQN reinforcement learning algorithm.

        Args:
            task (RLTask, Env): RL task/env to run.
            approximator (ParametrizedQValueOutput, PolicyFromQValue): approximator to use and update.
            gamma (float): discount factor (which is a bias-variance tradeoff). This parameter describes how much
                importance has the future rewards we get.
            lr (float): learning rate.
            capacity (int): capacity of the experience replay storage.
            num_workers (int): number of processes / workers to run in parallel.
        """
        # check given approximator
        if isinstance(approximator, ParametrizedQValueOutput):
            policy = PolicyFromQValue(approximator)
            q_value = approximator
        elif isinstance(approximator, PolicyFromQValue):
            policy = approximator
            q_value = approximator.value
        else:
            raise TypeError("Expecting the given approximator to be an instance of `PolicyFromQValue`, or "
                            "`ParametrizedQValueOutput`, instead got: {}".format(type(approximator)))

        # evaluate target Q-value fct by copying Q-value function approximator
        q_target = copy.deepcopy(q_value)

        # create action exploration strategy
        exploration = EpsilonGreedyActionExploration(policy=policy, action=policy.actions)

        # create experience replay
        storage = ExperienceReplay(capacity=capacity)

        # create target return estimator
        estimator = TDQLearningReturn(q_value=q_value, target_qvalue=q_target, gamma=gamma)

        # create loss
        loss = HuberLoss(MSBELoss(td_return=estimator), delta=1.)

        # create optimizer
        optimizer = Adam(learning_rate=lr)

        # create target updater
        target_updater = CopyParameter(sleep_count=100)  # PolyakAveraging(rho=0.5)

        # define the 3 main steps in RL: explore, evaluate, and update
        explorer = Explorer(task, exploration, storage, num_workers=num_workers)
        evaluator = Evaluator(estimator)
        updater = Updater(policy, storage, loss, optimizer, updaters={target_updater: q_target})

        # initialize RL algorithm
        super(DQN, self).__init__(explorer, evaluator, updater)
