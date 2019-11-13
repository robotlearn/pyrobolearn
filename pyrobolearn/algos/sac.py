#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Soft-Actor Critic algorithm.

Define the SAC reinforcement learning algorithm. This is a model-free, off-policy, actor-critic method.
"""

import copy
import collections

from pyrobolearn.algos.rl_algo import GradientRLAlgo, Explorer, Evaluator, Updater

from pyrobolearn.policies import Policy
from pyrobolearn.values import Value, QValue
from pyrobolearn.actorcritics import ActorCritic
from pyrobolearn.exploration import ActionExploration, GaussianActionExploration

from pyrobolearn.storages import ExperienceReplay
from pyrobolearn.samplers import BatchRandomSampler
from pyrobolearn.returns import ValueTarget, EntropyValueTarget
from pyrobolearn.losses import MSBELoss, QLoss
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


class SAC(GradientRLAlgo):
    r"""Soft Actor-Critic

    Type: policy-gradient, off-policy, continuous

    The following documentation is copied-pasted from [2], and is reproduced here for completeness. If you use
    this algorithm, please acknowledge / cite [1, 2].


    Background
    ----------

    "Soft Actor Critic (SAC) is an algorithm which optimizes a stochastic policy in an off-policy way, forming a
    bridge between stochastic policy optimization and DDPG-style approaches. It isn't a direct successor to TD3
    (having been published roughly concurrently), but it incorporates the clipped double-Q trick, and due to the
    inherent stochasticity of the policy in SAC, it also winds up benefiting from something like target policy
    smoothing.

    A central feature of SAC is entropy regularization. The policy is trained to maximize a trade-off between
    expected return and entropy, a measure of randomness in the policy. This has a close connection to the
    exploration-exploitation trade-off: increasing entropy results in more exploration, which can accelerate
    learning later on. It can also prevent the policy from prematurely converging to a bad local optimum" [2]


    Key Equations
    -------------

    "To explain Soft Actor Critic, we first have to introduce the entropy-regularized reinforcement learning setting.
    In entropy-regularized RL, there are slightly-different equations for value functions.

    * Entropy-Regularized Reinforcement Learning

    Entropy is a quantity which, roughly speaking, says how random a random variable is. If a coin is weighted so that
    it almost always comes up heads, it has low entropy; if it's evenly weighted and has a half chance of either
    outcome, it has high entropy.

    Let :math:`x` be a random variable with probability mass or density function :math:`P`. The entropy :math:`H` of
    :math:`x` is computed from its distribution :math:`P` according to

    .. math:: H(P) = \underE{x \sim P}{-\log P(x)}.

    In entropy-regularized reinforcement learning, the agent gets a bonus reward at each time step proportional to
    the entropy of the policy at that timestep. This changes the RL problem to:

    .. math::

        \pi^* = \arg \max_{\pi} \underE{\tau \sim \pi}{ \sum_{t=0}^{\infty} \gamma^t \bigg( R(s_t, a_t, s_{t+1}) +
                \alpha H\left(\pi(\cdot|s_t)\right) \bigg)},

    where :math:`\alpha > 0` is the trade-off coefficient. (Note: we're assuming an infinite-horizon discounted
    setting here, and we'll do the same for the rest of this page.) We can now define the slightly-different value
    functions in this setting. :math:`V^{\pi}` is changed to include the entropy bonuses from every timestep:

    .. math::

        V^{\pi}(s) = \underE{\tau \sim \pi}{ \left. \sum_{t=0}^{\infty} \gamma^t \bigg( R(s_t, a_t, s_{t+1}) +
                     \alpha H\left(\pi(\cdot|s_t)\right) \bigg) \right| s_0 = s}

    :math:`Q^{\pi}` is changed to include the entropy bonuses from every timestep except the first:

    .. math::

        Q^{\pi}(s,a) = \underE{\tau \sim \pi}{ \left. \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}) + \alpha
                       \sum_{t=1}^{\infty} \gamma^t H\left(\pi(\cdot|s_t)\right)\right| s_0 = s, a_0 = a}

    With these definitions, :math:`V^{\pi}` and :math:`Q^{\pi}` are connected by:

    .. math:: V^{\pi}(s) = \underE{a \sim \pi}{Q^{\pi}(s,a)} + \alpha H\left(\pi(\cdot|s)\right)

    and the Bellman equation for :math:`Q^{\pi}` is

    .. math::

        Q^{\pi}(s,a) &= \underE{s' \sim P \\ a' \sim \pi}{R(s,a,s') + \gamma\left(Q^{\pi}(s',a') + \alpha
                        H\left(\pi(\cdot|s')\right) \right)} \\ &= \underE{s' \sim P}{R(s,a,s') + \gamma V^{\pi}(s')}.

    * Soft Actor-Critic

    SAC concurrently learns a policy :math:`\pi_{\theta}`, two Q-functions :math:`Q_{\phi_1}`, :math:`Q_{\phi_2}`, and
    a value function :math:`V_{\psi}`.

    **Learning Q**: the Q-functions are learned by MSBE minimization, using a target value network to form the Bellman
    backups. They both use the same target, like in TD3, and have loss functions:

    .. math::

        L(\phi_i, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[ \Bigg( Q_{\phi_i}(s,a)
                                - \left(r + \gamma (1 - d) V_{\psi_{\text{targ}}}(s') \right) \Bigg)^2 \right].

    The target value network, like the target networks in DDPG and TD3, is obtained by polyak averaging the value
    network parameters over the course of training.

    **Learning V**: the value function is learned by exploiting (a sample-based approximation of) the connection
    between :math:`Q^{\pi}` and :math:`V^{\pi}`. Before we go into the learning rule, let's first rewrite the
    connection equation by using the definition of entropy to obtain:

    .. math::

        V^{\pi}(s) &= \underE{a \sim \pi}{Q^{\pi}(s,a)} + \alpha H\left(\pi(\cdot|s)\right) \\
                   &= \underE{a \sim \pi}{Q^{\pi}(s,a) - \alpha \log \pi(a|s)}.

    The RHS is an expectation over actions, so we can approximate it by sampling from the policy:

    .. math:: V^{\pi}(s) \approx Q^{\pi}(s,\tilde{a}) - \alpha \log \pi(\tilde{a}|s), \quad \tilde{a} \sim \pi(\cdot|s).

    SAC sets up a mean-squared-error loss for :math:`V_{\psi}` based on this approximation. But what Q-value do we use?
    SAC uses clipped double-Q like TD3 for learning the value function, and takes the minimum Q-value between the two
    approximators. So the SAC loss for value function parameters is:

    .. math:: L(\psi, {\mathcal D}) = \underE{s \sim \mathcal{D} \\ \tilde{a} \sim \pi_{\theta}}{\Bigg(V_{\psi}(s) -
                    \left(\min_{i=1,2} Q_{\phi_i}(s,\tilde{a}) - \alpha \log \pi_{\theta}(\tilde{a}|s) \right)\Bigg)^2}.

    Importantly, we do not use actions from the replay buffer here: these actions are sampled fresh from the current
    version of the policy.

    **Learning the Policy**: the policy should, in each state, act to maximize the expected future return plus
    expected future entropy. That is, it should maximize :math:`V^{\pi}(s)`, which we expand out (as before) into

    .. math:: \underE{a \sim \pi}{Q^{\pi}(s,a) - \alpha \log \pi(a|s)}.

    The way we optimize the policy makes use of the reparameterization trick, in which a sample from
    :math:`\pi_{\theta}(\cdot|s)` is drawn by computing a deterministic function of state, policy parameters, and
    independent noise. To illustrate: following the authors of the SAC paper, we use a squashed Gaussian policy,
    which means that samples are obtained according to

    .. math::

        \tilde{a}_{\theta}(s, \xi) = \tanh\left( \mu_{\theta}(s) + \sigma_{\theta}(s) \odot \xi \right), \quad
                                     \xi \sim \mathcal{N}(0, I).

    **Notice**:

    This policy has two key differences from the policies we use in the other policy optimization algorithms:

    1. The squashing function. The :math:`\tanh` in the SAC policy ensures that actions are bounded to a finite range.
    This is absent in the VPG, TRPO, and PPO policies. It also changes the distribution: before the :math:`\tanh` the
    SAC policy is a factored Gaussian like the other algorithms' policies, but after the :math:`\tanh` it is not. (You
    can still evaluate the log-probabilities of actions in closed form, though: see the paper appendix for details.)

    2. The way standard deviations are parameterized. In VPG, TRPO, and PPO, we represent the log std devs with
    state-independent parameter vectors. In SAC, we represent the log std devs as outputs from the neural network,
    meaning that they depend on state in a complex way. SAC with state-independent log std devs, in our experience,
    did not work. (Can you think of why? Or better yet: run an experiment to verify?)

    **End of notice**

    The reparameterization trick allows us to rewrite the expectation over actions (which contains a pain point:
    the distribution depends on the policy parameters) into an expectation over noise (which removes the pain point:
    the distribution now has no dependence on parameters):

    .. math::

        \underE{a \sim \pi_{\theta}}{Q^{\pi_{\theta}}(s,a) - \alpha \log \pi_{\theta}(a|s)}
        = \underE{\xi \sim \mathcal{N}}{Q^{\pi_{\theta}}(s,\tilde{a}_{\theta}(s,\xi))
        - \alpha \log \pi_{\theta}(\tilde{a}_{\theta}(s,\xi)|s)}

    To get the policy loss, the final step is that we need to substitute :math:`Q^{\pi_{\theta}}` with one of our
    function approximators. The same as in TD3, we use Q_{\phi_1}. The policy is thus optimized according to

    .. math::

        \max_{\theta} \underE{s \sim \mathcal{D} \\ \xi \sim \mathcal{N}}{Q_{\phi_1}(s,\tilde{a}_{\theta}(s,\xi))
            - \alpha \log \pi_{\theta}(\tilde{a}_{\theta}(s,\xi)|s)},

    which is almost the same as the DDPG and TD3 policy optimization, except for the stochasticity and entropy term."
    [2]


    Exploration vs. Exploitation
    ----------------------------

    "SAC trains a stochastic policy with entropy regularization, and explores in an on-policy way. The entropy
    regularization coefficient \alpha explicitly controls the explore-exploit tradeoff, with higher :math:`\alpha`
    corresponding to more exploration, and lower \alpha corresponding to more exploitation. The right coefficient
    (the one which leads to the stablest / highest-reward learning) may vary from environment to environment, and
    could require careful tuning.

    At test time, to see how well the policy exploits what it has learned, we remove stochasticity and use the mean
    action instead of a sample from the distribution. This tends to improve performance over the original stochastic
    policy." [2]


    Pseudo-algo
    -----------

    Pseudo-algorithm (taken from [2] and reproduce here for completeness)::
        1. Input: initial policy parameters :math:`\theta_0`, Q-function parameters :math:`\phi_1, \phi_2`,
            V-function parameters :math:`\psi`, empty replay buffer :math:`D`
        2. Set target parameters equal to main parameters :math:`\psi_{\text{target}} \leftarrow \psi`
        3. repeat:
        4.    Observe state :math:`s` and select action :math:`a \sim \pi_{\theta}(\cdot|s)`
        5.    Execute :math:`a` in the environment
        6.    Observe next state :math:`s'`, reward :math:`r`, and done signal :math:`d` to indicate whether :math:`s'`
              is terminal
        7.    Store :math:`(s, a, r, s', d)` in replay buffer :math:`D`
        8.    If :math:`s'` is terminal, reset environment state.
        9.    if it's time to update then:
        10.       for j in range(num_updates) do:
        11.           Randomly sample a batch of transitions, :math:`B = {(s, a, r, s', d)}` from :math:`D`
        12.           Compute targets for Q and V functions:
                        :math:`y_q(r,s',d) = r + \gamma (1-d) V_{\psi_{\text{targ}}}(s')`
                        :math:`y_v(s) = \min_{i=1,2} Q_{\phi_i} (s, \tilde{a}) - \alpha \log \pi_{\theta}(\tilde{a}|s)`,
                         with :math:`\tilde{a} \sim \pi_{\theta}(\cdot|s)`
        13.           Update Q-functions by one step of gradient descent using:
                        :math:`\nabla_{\phi_i} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi,i}(s,a) -
                        y_q(r,s',d) \right)^2`, for :math:`i=1,2`
        14.           Update V-function by one step of gradient descent using:
                        :math:`\nabla_{\psi} \frac{1}{|B|}\sum_{s \in B} \left( V_{\psi}(s) - y_v(s) \right)^2`
        15.           Update policy by one step of gradient ascent using:
                        :math:`\nabla_{\theta} \frac{1}{|B|}\sum_{s \in B} \Big( Q_{\phi,1}(s, \tilde{a}_{\theta}(s))
                        - \alpha \log \pi_{\theta} \left(\left. \tilde{a}_{\theta}(s) | s\right) \Big)`,
                        where :math:`\tilde{a}_{\theta}(s)` is a sample from :math:`\pi_{\theta}(\cdot|s)` which is
                        differentiable wrt :math:`\theta` via the reparametrization trick.
        16.           Update target value network with:
                        :math:`\psi_{\text{targ}} &\leftarrow \rho \psi_{\text{targ}} + (1-\rho) \psi`
        17.       end for
        18.   end if
        19. until convergence


    References:
        [1] "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
            Haarnoja et al, 2018
        [2] OpenAI - Spinning Up: https://spinningup.openai.com/en/latest/algorithms/sac.html
        [3] RLKit: https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py
    """

    def __init__(self, task, approximators, gamma=0.99, lr=5e-4, polyak=0.995, alpha=0.2, capacity=10000,
                 num_workers=1):
        """
        Initialize the SAC off-policy RL algorithm.

        Args:
            task (RLTask, Env): RL task/env to run
            approximators ([Policy, Value, QValue]): approximators to optimize.
            gamma (float): discount factor (which is a bias-variance tradeoff). This parameter describes how much
                importance has the future rewards we get.
            lr (float): learning rate
            polyak (float): coefficient (between 0 and 1) used in the polyak averaging when updating the target
                approximators. If 1, it will let the target parameter(s) unchanged, if 0 it will just copy the
                current parameter(s).
            alpha (float): entropy regularization coefficient which controls the tradeoff between exploration and
                exploitation. Higher :attr:`alpha` means more exploration, and lower :attr:`alpha` corresponds to more
                exploitation.
            capacity (int): capacity of the experience replay storage.
            num_workers (int): number of processes / workers to run in parallel
        """

        # check approximators
        if not isinstance(approximators, collections.Iterable):
            raise TypeError("Expecting the approximators to be a list containing a Policy, a Value, and at least 2 "
                            "QValues")
        policy, value, q_values = None, None, []
        for approximator in approximators:
            if isinstance(approximator, Policy):
                policy = approximator
            elif isinstance(approximator, Value):
                value = approximator
            elif isinstance(approximator, ActorCritic):
                policy = approximator.actor
                value = approximator.critic
            elif isinstance(approximator, QValue):
                q_values.append(approximator)

        if policy is None:
            raise TypeError("No policy was given to the algorithm.")
        if value is None:
            raise TypeError("No value function approximator was given to the algorithm.")
        if len(q_values) == 0:
            raise TypeError("No Q-value function approximators were given to the algorithm.")

        # set target parameters equal to main parameters for the value function
        value_target = copy.deepcopy(value, memo={})

        # create experience replay
        states, actions = policy.states, policy.actions
        storage = ExperienceReplay(state_shapes=states.merged_shape, action_shapes=actions.merged_shape,
                                   capacity=capacity)
        sampler = BatchRandomSampler(storage)

        # create action exploration
        exploration = ActionExploration(policy)

        # create targets
        q_target = ValueTarget(values=value_target, gamma=gamma)
        v_target = EntropyValueTarget(q_values=q_values, policy=exploration, alpha=alpha)

        # create losses
        q_loss = MSBELoss(td_return=estimator)
        policy_loss = QLoss(q_value=q_values[0], policy=policy)  # only the first q-value is used to train the policy
        losses = [q_loss, policy_loss]

        # create optimizer
        optimizer = Adam(learning_rate=lr)

        # create parameter updater for target value function
        params_updater = PolyakAveraging(current=value, target=value_target, rho=polyak)

        # define the 3 main steps in RL: explore, evaluate, and update
        explorer = Explorer(task, exploration, storage, num_workers=num_workers)
        evaluator = Evaluator(None)  # off-policy
        updater = Updater(approximators, sampler, losses, optimizer, updaters=params_updater)

        # initialize RL algorithm
        super(SAC, self).__init__(explorer, evaluator, updater)
