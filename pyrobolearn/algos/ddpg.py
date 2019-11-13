#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the  Deep Deterministic Policy Gradient (DDPG).

For the Twin Delayed DDPG algorithm, see `pyrobolearn/algos/td3.py`
"""

import copy

from pyrobolearn.algos.rl_algo import GradientRLAlgo, Explorer, Evaluator, Updater

from pyrobolearn.policies import Policy
from pyrobolearn.values import QValue
from pyrobolearn.exploration import ActionExploration, GaussianActionExploration

from pyrobolearn.storages import ExperienceReplay
from pyrobolearn.samplers import BatchRandomSampler
from pyrobolearn.returns import TDQValueReturn, QValueTarget
from pyrobolearn.losses import MSBELoss, QLoss, L2Loss, ValueL2Loss
from pyrobolearn.optimizers import Adam

from pyrobolearn.parameters.updater import PolyakAveraging


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse", "OpenAI Spinning Up (Josh Achiam)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DDPG(GradientRLAlgo):
    r"""Deep Deterministic Policy Gradients (DDPG)

    Type:: actor-critic method, off-policy, continuous action space, exploration in action-space (thus step-based).

    The documentation has been copied-pasted from [3], and is reproduced here for completeness. If you use this
    algorithm, please acknowledge / cite [1, 3].


    Background
    ----------

    "Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy.
    It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the
    policy.

    This approach is closely connected to Q-learning, and is motivated the same way: if you know the optimal
    action-value function :math:`Q^*(s,a)`, then in any given state, the optimal action :math:`a^*(s)` can be found by
    solving:

    .. math:: a^*(s) = \arg \max_a Q^*(s,a).

    DDPG interleaves learning an approximator to :math:`Q^*(s,a)` with learning an approximator to :math:`a^*(s)`, and
    it does so in a way which is specifically adapted for environments with continuous action spaces. But what does it
    mean that DDPG is adapted specifically for environments with continuous action spaces? It relates to how we
    compute the max over actions in :math:`\max_a Q^*(s,a)`.

    When there are a finite number of discrete actions, the max poses no problem, because we can just compute the
    Q-values for each action separately and directly compare them. (This also immediately gives us the action which
    maximizes the Q-value.) But when the action space is continuous, we can't exhaustively evaluate the space, and
    solving the optimization problem is highly non-trivial. Using a normal optimization algorithm would make
    calculating :math:`\max_a Q^*(s,a)` a painfully expensive subroutine. And since it would need to be run every
    time the agent wants to take an action in the environment, this is unacceptable.

    Because the action space is continuous, the function :math:`Q^*(s,a)` is presumed to be differentiable with
    respect to the action argument. This allows us to set up an efficient, gradient-based learning rule for a policy
    :math:`\mu(s)` which exploits that fact. Then, instead of running an expensive optimization subroutine each time
    we wish to compute :math:`\max_a Q(s,a)`, we can approximate it with :math:`\max_a Q(s,a) \approx Q(s,\mu(s))`.
    See the Key Equations section details." [3]


    Key Equations
    -------------

    "Here are the math behind the two parts of DDPG: learning a Q function, and learning a policy.

    * The Q-Learning Side of DDPG

    First, let's recap the Bellman equation describing the optimal action-value function, Q^*(s,a). It's given by

    .. math:: Q^*(s,a) = \underset{s' \sim P}{{\mathrm E}}\left[r(s,a) + \gamma \max_{a'} Q^*(s', a')\right]

    where :math:`s' \sim P` is shorthand for saying that the next state, :math:`s'`, is sampled by the environment
    from a distribution :math:`P(\cdot| s, a)`.

    This Bellman equation is the starting point for learning an approximator to :math:`Q^*(s,a)`. Suppose the
    approximator is a neural network :math:`Q_{\phi}(s,a)`, with parameters :math:`\phi`, and that we have collected
    a set :math:`D` of transitions :math:`(s, a, r, s', d)` (where :math:`d` indicates whether state :math:`s'` is
    terminal). We can set up a mean-squared Bellman error (MSBE) function, which tells us roughly how closely
    :math:`Q_{\phi}` comes to satisfying the Bellman equation:

    .. math:: L(\phi, D) = \underset{(s,a,r,s',d) \sim D}{{\mathrm E}}\left[ \Bigg( Q_{\phi}(s,a) - \left(r + \gamma
                            (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2 \right]

    Here, in evaluating :math:`(1-d)`, we've used a Python convention of evaluating True to 1 and False to zero. Thus,
    when d==True - which is to say, when :math:`s'` is a terminal state - the Q-function should show that the agent
    gets no additional rewards after the current state.

    Q-learning algorithms for function approximators, such as DQN (and all its variants) and DDPG, are largely based
    on minimizing this MSBE loss function. There are two main tricks employed by all of them which are worth
    describing, and then a specific detail for DDPG.

    1. Trick One: Replay Buffers. All standard algorithms for training a deep neural network to approximate
    :math:`Q^*(s,a)` make use of an experience replay buffer. This is the set :math:`D` of previous experiences.
    In order for the algorithm to have stable behavior, the replay buffer should be large enough to contain a wide
    range of experiences, but it may not always be good to keep everything. If you only use the very-most recent data,
    you will overfit to that and things will break; if you use too much experience, you may slow down your learning.
    This may take some tuning to get right.

    **Notice**:

    We've mentioned that DDPG is an off-policy algorithm: this is as good a point as any to highlight why and how.
    Observe that the replay buffer should contain old experiences, even though they might have been obtained using an
    outdated policy. Why are we able to use these at all? The reason is that the Bellman equation doesn't care which
    transition tuples are used, or how the actions were selected, or what happens after a given transition, because
    the optimal Q-function should satisfy the Bellman equation for all possible transitions. So any transitions that
    we've ever experienced are fair game when trying to fit a Q-function approximator via MSBE minimization.

    **End of notice**

    2. Trick Two: Target Networks. Q-learning algorithms make use of target networks. The term

    .. math:: r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a')

    is called the target, because when we minimize the MSBE loss, we are trying to make the Q-function be more like
    this target. Problematically, the target depends on the same parameters we are trying to train: :math:`\phi`.
    This makes MSBE minimization unstable. The solution is to use a set of parameters which comes close to
    :math:`\phi`, but with a time delay - that is to say, a second network, called the target network, which lags the
    first. The parameters of the target network are denoted :math:`\phi_{\text{targ}}`.

    In DQN-based algorithms, the target network is just copied over from the main network every some-fixed-number of
    steps. In DDPG-style algorithms, the target network is updated once per main network update by polyak averaging:

    .. math:: \phi_{\text{targ}} \leftarrow \rho \phi_{\text{targ}} + (1 - \rho) \phi,

    where :math:`\rho` is a hyperparameter between 0 and 1 (usually close to 1).

    **DDPG Detail: Calculating the Max Over Actions in the Target**. As mentioned earlier: computing the maximum over
    actions in the target is a challenge in continuous action spaces. DDPG deals with this by using a target policy
    network to compute an action which approximately maximizes :math:`Q_{\phi_{\text{targ}}}`. The target policy
    network is found the same way as the target Q-function: by polyak averaging the policy parameters over the course
    of training.

    Putting it all together, Q-learning in DDPG is performed by minimizing the following MSBE loss with stochastic
    gradient descent:

    .. math:: L(\phi, D) = \underset{(s,a,r,s',d) \sim D}{{\mathrm E}}\left[ \Bigg( Q_{\phi}(s,a) - \left(r + \gamma
                            (1 - d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s')) \right) \Bigg)^2 \right],

    where :math:`\mu_{\theta_{\text{targ}}}` is the target policy.


    * The Policy Learning Side of DDPG

    Policy learning in DDPG is fairly simple. We want to learn a deterministic policy :math:`\mu_{\theta}(s)` which
    gives the action that maximizes :math:`Q_{\phi}(s,a)`. Because the action space is continuous, and we assume the
    Q-function is differentiable with respect to action, we can just perform gradient ascent (with respect to policy
    parameters only) to solve

    .. math:: \max_{\theta} \underset{s \sim D}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right].

    Note that the Q-function parameters are treated as constants here." [3]


    Exploration vs. Exploitation
    ----------------------------

    "DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the agent were
    to explore on-policy, in the beginning it would probably not try a wide enough variety of actions to find useful
    learning signals. To make DDPG policies explore better, we add noise to their actions at training time.
    The authors of the original DDPG paper recommended time-correlated OU noise, but more recent results suggest that
    uncorrelated, mean-zero Gaussian noise works perfectly well. Since the latter is simpler, it is preferred.
    To facilitate getting higher-quality training data, you may reduce the scale of the noise over the course of
    training.

    At test time, to see how well the policy exploits what it has learned, we do not add noise to the actions." [3]


    Pseudo-algo
    -----------

    Pseudo-algorithm (taken from [3] and reproduce here for completeness)::
        1. Input: initial policy parameters :math:`\theta_0`, initial Q-value function parameters :math:`\phi_0`,
            empty replay buffer :math:`D`.
        2. Set target parameters equal to main parameters :math:`\theta_{\text{targ}} \leftarrow \theta`,
            :math:`\phi_{\text{targ}} \leftarrow \phi`
        3. repeat:
        4.     Observe state :math:`s` and select action :math:`a = \text{clip}(\mu_{\theta}(s) + \epsilon, a_{Low},
                a_{High})`, where :math:`\epsilon \sim \mathcal{N}`
        5.     Execute :math:`a` in the environment
        6.     Observe next state :math:`s'`, reward :math:`r`, and done signal :math:`d` to indicate whether
                :math:`s'` is terminal.
        7.     Store :math:`(s, a, r, s', d)` in replay buffer :math:`D`
        8.     If :math:`s'` is terminal, reset environment state.
        9.     if it's time to update then:
        10.       for j in range(num_updates) do:
        11.           Randomly sample a batch of transitions, :math:`B = {(s, a, r, s', d)}` from :math:`D`
        12.           Compute targets
                        :math:`y(r,s',d) = r + \gamma (1-d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s'))`
        13.           Update Q-function by one step of gradient descent using
                        :math:`\nabla_{\phi} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} ( Q_{\phi}(s,a) - y(r,s',d) )^2`
        14.           Update policy by one step of gradient ascent using
                        :math:`\nabla_{\theta} \frac{1}{|B|}\sum_{s \in B} Q_{\phi}(s, \mu_{\theta}(s))`
        15.           Update target networks with
                        :math:`\phi_{\text{targ}} \leftarrow \rho \phi_{\text{targ}} + (1-\rho) \phi`
                        :math:`\theta_{\text{targ}} \leftarrow \rho \theta_{\text{targ}} + (1-\rho) \theta`
        16.       end for
        17.    end if
        18. until convergence


    .. seealso:: For the "Twin Delayed Deep Deterministic Policy Gradients", see `pyrobolearn/algos/td3.py`.


    References:
        [1] "Deterministic Policy Gradient Algorithm", Silver et al., 2014
        [2] "Continuous Control with Deep Reinforcement Learning", Lillicrap et al., 2015
        [3] OpenAI - Spinning Up: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        [4] PyTorch implementation by Kostrikov: https://github.com/ikostrikov/pytorch-ddpg-naf
        [5] "Policy Gradient Algorithms":
            https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
    """

    def __init__(self, task, approximators, gamma=0.99, lr=0.001, polyak=0.995, capacity=10000, num_workers=1):
        """
        Initialize the DDPG off-policy RL algorithm.

        Args:
            task (RLTask, Env): RL task/env to run
            approximators ([Policy, QValue]): policy and Q-value function approximator to optimize.
            gamma (float): discount factor (which is a bias-variance trade-off). This parameter describes how much
                importance has the future rewards we get.
            lr (float): learning rate
            polyak (float): coefficient (between 0 and 1) used in the polyak averaging when updating the target
                approximators. If 1, it will let the target parameter(s) unchanged, if 0 it will just copy the
                current parameter(s).
            capacity (int): capacity of the experience replay storage.
            num_workers (int): number of processes / workers to run in parallel
        """

        # check given approximators
        if isinstance(approximators, (tuple, list)) and len(approximators) != 2:

            # get the policy and Q-value approximator
            policy, q_value = None, None
            for approximator in approximators:
                if isinstance(approximator, (Policy, QValue)):
                    policy = approximator
                elif isinstance(approximator, QValue):
                    q_value = approximator

            # check that the policy and Q-value approximator are different than None
            if policy is None:
                raise ValueError("No policy approximator was given to the algorithm.")
            if q_value is None:
                raise ValueError("No Q-value approximator was given to the algorithm.")

        else:
            raise TypeError("Expecting a list/tuple of a policy and a Q-value function.")

        # get states and actions from policy
        states, actions = policy.states, policy.actions

        # check that the actions are continuous
        if not actions.is_continuous():
            raise ValueError("The DDPG assumes that the actions are continuous, however got an action which is not.")

        # Set target parameters equal to main parameters
        memo = {}
        q_target = copy.deepcopy(q_value, memo=memo)
        policy_target = copy.deepcopy(policy, memo=memo)

        # create action exploration strategy
        exploration = ActionExploration(policy=policy, action=actions)

        # create experience replay
        storage = ExperienceReplay(state_shapes=states.merged_shape, action_shapes=actions.merged_shape,
                                   capacity=capacity)
        sampler = BatchRandomSampler(storage)

        # create target return estimator
        # target = QValueTarget(q_values=q_target, policy=policy_target, gamma=gamma)
        returns = TDQValueReturn(q_value=q_value, policy=policy_target, target_qvalue=q_target, gamma=gamma)

        # create Q-value loss and policy loss
        # q_loss = L2Loss(target=target, predictor=q_value)
        # q_loss = ValueLoss(returns=target, value=q_value)
        q_loss = MSBELoss(td_return=returns)
        policy_loss = QLoss(q_value=q_value, policy=policy)
        losses = [q_loss, policy_loss]

        # create optimizer
        optimizer = Adam(learning_rate=lr)

        # create q value and policy updaters
        q_value_updater = PolyakAveraging(current=q_value, target=q_target, rho=polyak)
        policy_updater = PolyakAveraging(current=policy, target=policy_target, rho=polyak)

        # define the 3 main steps in RL: explore, evaluate, and update
        explorer = Explorer(task, exploration, storage, num_workers=num_workers)
        evaluator = Evaluator(None)  # off-policy
        updater = Updater(approximators, sampler, losses, optimizer, evaluators=returns,
                          updaters=[q_value_updater, policy_updater])

        # initialize RL algorithm
        super(DDPG, self).__init__(explorer, evaluator, updater)
