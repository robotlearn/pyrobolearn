#!/usr/bin/env python
"""Provide the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

The REINFORCE/VPG is a model-free, off-policy, gradient policy-based algorithm that works only for continuous actions.
"""

import copy

from pyrobolearn.algos.rl_algo import GradientRLAlgo, Explorer, Evaluator, Updater

from pyrobolearn.policies import Policy
from pyrobolearn.values import QValue
from pyrobolearn.exploration import ActionExploration, GaussianActionExploration

from pyrobolearn.storages import ExperienceReplay
from pyrobolearn.samplers import BatchRandomSampler
from pyrobolearn.estimators import TDQValueReturn
from pyrobolearn.losses import MSBELoss, QLoss
from pyrobolearn.optimizers import Adam

from pyrobolearn.parameters.updater import PolyakAveraging


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse", "OpenAI Spinning Up (Josh Achiam)"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TD3(GradientRLAlgo):
    r"""Twin Delayed Deep Deterministic Policy Gradient

    Type:: model-free, off-policy, gradient-based actor-critic algorithm for continuous only action spaces.

    The documentation has been copied-pasted from [2], and is reproduced here for completeness. If you use this
    algorithm, please acknowledge / cite [1, 2].


    Background
    ----------

    "While DDPG can achieve great performance sometimes, it is frequently brittle with respect to hyperparameters and
    other kinds of tuning. A common failure mode for DDPG is that the learned Q-function begins to dramatically
    overestimate Q-values, which then leads to the policy breaking, because it exploits the errors in the Q-function.
    Twin Delayed DDPG (TD3) is an algorithm which addresses this issue by introducing three critical tricks:

    1. Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence 'twin'), and uses the
    smaller of the two Q-values to form the targets in the Bellman error loss functions.
    2. Trick Two: 'Delayed' Policy Updates. TD3 updates the policy (and target networks) less frequently than the
    Q-function. The paper recommends one policy update for every two Q-function updates.
    3. Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy
    to exploit Q-function errors by smoothing out Q along changes in action.

    Together, these three tricks result in substantially improved performance over baseline DDPG." [2]


    Key Equations
    -------------

    "TD3 concurrently learns two Q-functions, :math:`Q_{\phi_1}` and :math:`Q_{\phi_2}`, by mean square Bellman error
    minimization, in almost the same way that DDPG learns its single Q-function. To show exactly how TD3 does this
    and how it differs from normal DDPG, we'll work from the innermost part of the loss function outwards.

    First: target policy smoothing. Actions used to form the Q-learning target are based on the target policy,
    :math:`\mu_{\theta_{\text{targ}}}`, but with clipped noise added on each dimension of the action.
    After adding the clipped noise, the target action is then clipped to lie in the valid action range (all valid
    actions, :math:`a`, satisfy :math:`a_{Low} \leq a \leq a_{High}`). The target actions are thus:

    .. math:: a'(s') = \text{clip}\left(\mu_{\theta_{\text{targ}}}(s') + \text{clip}(\epsilon,-c,c), a_{Low},
                        a_{High}\right), \qquad \epsilon \sim \mathcal{N}(0, \sigma)

    Target policy smoothing essentially serves as a regularizer for the algorithm. It addresses a particular failure
    mode that can happen in DDPG: if the Q-function approximator develops an incorrect sharp peak for some actions,
    the policy will quickly exploit that peak and then have brittle or incorrect behavior. This can be averted by
    smoothing out the Q-function over similar actions, which target policy smoothing is designed to do.

    Next: clipped double-Q learning. Both Q-functions use a single target, calculated using whichever of the two
    Q-functions gives a smaller target value:

    .. math:: y(r,s',d) = r + \gamma (1 - d) \min_{i=1,2} Q_{\phi_{i, \text{targ}}}(s', a'(s')),

    and then both are learned by regressing to this target:

    .. math::

        L(\phi_1,{\mathcal D}) = \underE{(s,a,r,s',d) \sim {\mathcal D}}{\left(Q_{\phi_1}(s,a) - y(r,s',d) \right)^2},

        L(\phi_2,{\mathcal D}) = \underE{(s,a,r,s',d) \sim {\mathcal D}}{\left(Q_{\phi_2}(s,a) - y(r,s',d) \right)^2 }.

    Using the smaller Q-value for the target, and regressing towards that, helps fend off overestimation in the
    Q-function.

    Lastly: the policy is learned just by maximizing Q_{\phi_1}:

    .. math:: \max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi_1}(s, \mu_{\theta}(s)) \right],

    which is pretty much unchanged from DDPG. However, in TD3, the policy is updated less frequently than the
    Q-functions are. This helps damp the volatility that normally arises in DDPG because of how a policy update changes
    the target." [2]


    Exploration vs. Exploitation
    ----------------------------

    "TD3 trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the agent were to
    explore on-policy, in the beginning it would probably not try a wide enough variety of actions to find useful
    learning signals. To make TD3 policies explore better, we add noise to their actions at training time, typically
    uncorrelated mean-zero Gaussian noise. To facilitate getting higher-quality training data, you may reduce the
    scale of the noise over the course of training. (We do not do this in our implementation, and keep noise scale
    fixed throughout.)

    At test time, to see how well the policy exploits what it has learned, we do not add noise to the actions." [2]


    Pseudo-algo:
    -----------

    Pseudo-algorithm (taken from [2] and reproduce here for completeness)::
        1. Input: initial policy parameters :math:`\theta`, Q-function parameters :math:`\phi_1, \phi_2`, empty replay
            buffer :math:`D`
        2. Set target parameters equal to main parameters :math:`\theta_{target} \leftarrow \theta`,
            :math:`\phi_{target, i} \leftarrow \phi_i` for i=1,2
        3. repeat:
        4.    Observe state :math:`s` and select action :math:`a = clip(\mu_{\theta}(s) + \epsilon, a_{Low}, a_{High})`,
              where :math:`\epsilon \sim \mathcal{N}`
        5.    Execute :math:`a` in the environment
        6.    Observe next state :math:`s'`, reward :math:`r`, and done signal :math:`d` to indicate whether :math:`s'`
              is terminal.
        7.    Store :math:`(s, a, r, s', d)` in replay buffer :math:`D`
        8.    If :math:`s'` is terminal, reset environment state.
        9.    if it's time to update then:
        10.      for j in range(num_updates) do:
        11.          Randomly sample a batch of transitions, :math:`B = {(s, a, r, s', d)}` from :math:`D`
        12.          Compute target actions
                       :math:`a'(s') = clip(\mu_{\theta_target}(s') + clip(\epsilon, -c, c), a_{Low}, a_{High})`,
                       where :math:`\epsilon \sim \mathcal{N}`
        13.          Compute targets
                       :math:`y(r,s',d) = r + \gamma (1 - d) \min_{i=1,2} Q_{\phi_{target, i}}(s', a'(s))`
        14.          Update Q-functions by one step of gradient descent using
                       :math:`\grad_{\phi_i} \frac{1}{|B|} \sum_{(s,a,r,s',d) \in B} (Q_{\phi_i}(s,a) - y(r,s',d))^2`,
                       for i=1,2
        15.          if (j % policy_delay) == 0 then:
        16.             Update policy by one step of gradient ascent using
                         :math:`\grad_{\theta} \frac{1}{|B|} \sum_{s \in B} Q_{\phi_1}(s, \mu_{\theta}(s))`
        17.             Update target networks with
                         :math:`\phi_{target,i} \leftarrow \rho \phi_{target,i} + (1 - \rho)\phi_i` for i=1,2
                         :math:`\theta_{target} \leftarrow \rho \theta_{target} + (1 - \rho) \theta`
        18.          end if
        19.      end for
        20.   end if
        21. until convergence


    References::
        [1] "Addressing Function Approximation Error in Actor-Critic Methods", Fujimoto et al., 2018
        [2] OpenAI - Spinning Up: https://spinningup.openai.com/en/latest/algorithms/td3.html
    """

    def __init__(self, task, approximators, gamma=0.99, lr=0.001, polyak=0.995, capacity=10000, num_workers=1):
        """
        Initialize the TD3 off-policy RL algorithm.

        Args:
            task (RLTask, Env): RL task/env to run
            approximators (Policy, [Policy, Value], ActorCritic): approximators to optimize
            gamma (float): discount factor (which is a bias-variance tradeoff). This parameter describes how much
                importance has the future rewards we get.
            lr (float): learning rate.
            polyak (float): coefficient in the polyak averaging when updating the target approximators.
            capacity (int): capacity of the experience replay storage.
            num_workers (int): number of processes / workers to run in parallel
        """

        # check given approximators
        if isinstance(approximators, (tuple, list)):

            # get the policy and Q-value approximator
            policy, q_values = None, []
            for approximator in approximators:
                if isinstance(approximator, (Policy, QValue)):
                    policy = approximator
                elif isinstance(approximator, QValue):
                    q_values.append(approximator)

            # check that the policy and Q-value approximator are different than None
            if policy is None:
                raise ValueError("No policy approximator was given to the algorithm.")
            if not q_values:
                raise ValueError("No Q-value approximator was given to the algorithm.")

        else:
            raise TypeError("Expecting a list/tuple of a policy and a Q-value functions.")

        # check that there is at least 2 Q-value function approximators (the user can have more)
        if len(q_values) < 2:
            raise ValueError("Expecting at least 2 Q-value function approximators for the TD3 algorithm.")

        # check that the actions are continuous
        actions = policy.actions
        if not actions.is_continuous():
            raise ValueError("The TD3 assumes that the actions are continuous, however got an action which is not.")

        # evaluate target Q-value fct by copying Q-value function approximator
        q_targets = [copy.deepcopy(q_value) for q_value in q_values]
        policy_target = copy.deepcopy(policy)

        # create action exploration strategy
        exploration = ActionExploration(policy=policy, action=policy.actions)

        # create experience replay
        storage = ExperienceReplay(observation_shapes=policy.states, action_shapes=policy.actions, capacity=capacity)
        sampler = BatchRandomSampler(storage)

        # create target return estimator
        estimator = TDQValueReturn(q_value=q_values, policy=policy_target, target_qvalue=q_targets, gamma=gamma)

        # create Q-value loss and policy loss
        q_loss = MSBELoss(td_return=estimator)
        policy_loss = QLoss(q_value=q_values[0], policy=policy)  # only the first q-value is used to train the policy
        losses = [q_loss, policy_loss]

        # create optimizer
        optimizer = Adam(learning_rate=lr)

        # create q value and policy updaters
        params_updater = PolyakAveraging(rho=polyak)
        params_updaters = [(params_updater, q_target) for q_target in q_targets] + [(params_updater, policy_target)]

        # define the 3 main steps in RL: explore, evaluate, and update
        explorer = Explorer(task, exploration, storage, num_workers=num_workers)
        evaluator = Evaluator(estimator)
        updater = Updater(approximators, sampler, losses, optimizer, params_updaters)

        # initialize RL algorithm
        super(TD3, self).__init__(explorer, evaluator, updater)


# alias
TDDDPG = TD3
