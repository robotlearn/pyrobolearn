#!/usr/bin/env python
"""Provide the REINFORCE/VPG algorithm.

The REINFORCE/VPG is a model-free, on-policy, gradient policy-based algorithm.
"""

# from pyrobolearn.envs import Env
from pyrobolearn.policies import Policy
# from pyrobolearn.tasks import RLTask
from pyrobolearn.algos.rl_algo import GradientRLAlgo, Explorer, Evaluator, Updater

from pyrobolearn.values import ValueApproximator
from pyrobolearn.actorcritics import ActorCritic

from pyrobolearn.exploration import ActionExploration

from pyrobolearn.storages import RolloutStorage
from pyrobolearn.samplers import StorageSampler
from pyrobolearn.returns import ActionRewardEstimator, PolicyEvaluator
from pyrobolearn.losses import PGLoss, ValueL2Loss
from pyrobolearn.optimizers import Adam


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class REINFORCE(GradientRLAlgo):
    r"""REINFORCE: REward Increment = Nonnegative Factor * Offset Reinforcement * Characteristic Eligibility

    Type:: model-free, on-policy, gradient-based policy search algorithm for discrete or continuous action spaces.

    This class implements the REINFORCE (aka Vanilla Policy Gradient (VPG)) algorithm. This was the first
    policy-gradient method invented that uses the likelihood ratio trick.
    "The key idea underlying policy gradients is to push up the probabilities of actions that lead to higher return,
    and push down the probabilities of actions that lead to lower return, until you arrive at the optimal policy." [5]


    Mathematics
    -----------

    The goal in reinforcement learning is to maximize the expected return over all the possible trajectories:

    .. math::

        J(\theta) &= \int_{\mathbb{T}} p_{\theta}(\tau) R(\tau) d\tau \\
                  &= \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[R(\tau)]

    with

    .. math::

        p_{\theta}(\tau) = p(s_0) \prod_{t=0}^{T} p(s_{t+1} | s_t, a_t) \pi_{\theta}(a_t | s_t)

        R_{\tau} = \frac{1}{T} \sum_{t=0}^{T} c_t r(s_t, a_t, s_{t+1})

    where :math:`\pi` represents the policy (i.e. :math:`p(a_t|s_t)`) over which we have control,
    :math:`\theta` represents the policy parameters (that can be learned/optimized),
    :math:`\tau` is a trajectory (i.e. :math:`tau = (s_0, a_0, s_1,..., a_{T-1}, s_T)`,
    :math:`\mathbb{T}` represents the set of all possible trajectories,
    :math:`p(s_{t+1} | s_t, a_t)` is the dynamic model (aka the transition probability) which is determined
    by the environment (and thus, we don't have any control over it),
    :math:`R(\tau)` is the reward associated with the trajectory, :math:`r` is the instantaneous reward, and
    :math:`c_t` is a weighting coefficient.

    By taking the gradient of this objective with respect to the parameters :math:`\theta` and using the likelihood
    ratio trick :math:`\nabla_{\theta} p_{\theta}(\tau) = p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau)`, we
    can write:

    .. math::

        \nabla_{\theta} J(\theta) = \int_{\mathbb{T}} p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau)R(\tau)d\tau
                                  = \mathbb{E}_{\tau \sim p_{\theta}(\tau)} [ \nabla_{\theta}p_{\theta}(\tau) R(\tau) ]
                                  = \mathbb{E}[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) R(\tau) ]

    We can reduce the variance of the above gradient without introducing any biases by first noticing that actions
    at a certain step do not affect previous rewards. Second, removing . Thus, the gradient for REINFORCE can be
    rewritten as:

    ..math::

        g = \mathbb{E}[ (\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t))
                        (\sum_{t'=t}^{T} c_t r(s_t, a_t, s_{t+1}) - b) ]

    where :math:`b` is the baseline. The optimal baseline can be computed using the value function :math:`V(s)`.

    Because the integral over all the trajectories is impractical in practice, we instead make use of Monte Carlo
    expectations.


    Properties
    ----------

    Properties:
        - the expectation is guaranteed to converge to the true gradient [2]
    Pros:
        - easy to implement
        - basic baseline that can be used when comparing with other algorithms
    Cons:
        - requires several samples to estimate correctly the gradient
        - the learning can be quite unstable; the algorithm is pretty sensible to the value of the learning rate. This
          can result in a policy that can vary a lot during the training.

    Notes::
    * We can use off-policy exploration using importance sampling.


    Pseudo-algo
    -----------

    Pseudo-algorithm (taken from [5] and reproduced here for completeness)::
        1. Input: initial policy parameters :math:`\theta_0`, initial value function parameters :math:`\phi_0`
        2. for k=0,1,...,num_episodes do
        3.     Exploration: Collect set of trajectories :math:`D_k=\{\tau_i\}` by running policy :math:`\pi_{\theta_k}`
                in the environment.
        4.     Evaluation: Compute rewards-to-go :math:`\hat{R}_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})`,
                evaluate advantage estimates :math:`\hat{A}_t` (using any method of advantage estimation) based on the
                current value function :math:`V_{\phi_k}`.
        5.     Update:
                - Update the policy by maximizing the PG objective (using e.g. gradient ascent):
            :math:`\theta_{k+1} = \argmax_\theta \frac{1}{|D_k|T} \sum_{\tau \in D_k} \sum_{t=0}^T
               \pi_{\theta_k}(a_t | s_t) A^{\pi_{\theta_k}(s_t, a_t)}`
                - Update the value function by regression on the mean-squared error (using e.g. gradient descent):
            :math:`\phi_{k+1} = \argmin_\phi \frac{1}{|D_k|T} \sum_{\tau \in D_k} \sum_{t=0}^T
               (V_{\phi_k}(s_t) - \hat{R}_t)^2`


    References:
        [1] "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", Williams, 1992
        [2] "Policy Gradient Methods", Peters, 2010 (Scholarpedia)
        [3] "A Survey on Policy Search for Robotics", Deisenroth et al., 2013
        [4] PyTorch Reinforce: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
        [5] OpenAI - Spinning Up: https://spinningup.openai.com/en/latest/algorithms/vpg.html
        [6] "Policy Gradient Algorithms":
            https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

    Other implementations:
    - https://github.com/rll/rllab/blob/master/rllab/algos/vpg.py
    - https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    - https://github.com/JamesChuanggg/pytorch-REINFORCE
    - schulman's presentation
    - SLM-LAB: https://github.com/kengz/SLM-Lab
    - https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/3-reinforce/cartpole_reinforce.py
    """

    def __init__(self, task, approximators, gamma=0.99, lr=0.001, num_workers=1):
        """
        Initialize the REINFORCE on-policy RL algorithm.

        Args:
            task (RLTask, Env): RL task/env to run
            approximators (Policy, [Policy, Value], ActorCritic): approximators to optimize
            gamma (float): discount factor (which is a bias-variance tradeoff). This parameter describes how much
                importance has the future rewards we get.
            lr (float): learning rate
            num_workers (int): number of processes / workers to run in parallel
        """

        # check approximators
        policy, value, actor_critic = None, None, None
        if isinstance(approximators, Policy):
            policy = approximators
            if not policy.is_parametric():
                raise ValueError("The policy should be parametric.")
        elif isinstance(approximators, (tuple, list)):
            for approximator in approximators:
                if isinstance(approximator, Policy):
                    policy = approximator
                elif isinstance(approximator, ValueApproximator):
                    value = approximator
            actor_critic = ActorCritic(policy, value)
        elif isinstance(approximators, ActorCritic):
            policy = approximators.actor
            value = approximators.critic
            actor_critic = approximators
        else:
            raise TypeError("Expecting the approximators to be an instance of `Policy`, or `ActorCritic`, instead got:"
                            " {}".format(type(approximators)))

        # create exploration strategy (if action is discrete, boltzmann exploration. If action is continuous, gaussian)
        exploration = ActionExploration(policy)

        # create storage
        states, actions = policy.states, policy.actions
        storage = RolloutStorage(num_steps=1000, state_shapes=states.shape, action_shapes=actions.shape,
                                 num_trajectories=1)
        sampler = StorageSampler(storage)

        # create return: R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}
        returns = ActionRewardEstimator(storage, gamma=gamma)

        # create policy evaluator that will compute :math:`a \sim \pi(.|s_t)` and :math:`\pi(.|s_t)` on batch
        policy_evaluator = PolicyEvaluator(policy=exploration)

        # create loss for policy: \mathbb{E}[ \log \pi_{\theta}(a_t | s_t) R_t ]
        loss = PGLoss(returns)

        # create optimizer for policy (and possibly value function)
        optimizer = Adam(learning_rate=lr)

        # if value function, create its loss
        if value is not None:
            approximators = [policy, value]
            value_loss = ValueL2Loss(returns, value)
            loss = [loss, value_loss]
        else:
            approximators = policy

        # define the 3 main steps in RL: explore, evaluate, and update
        explorer = Explorer(task, exploration, storage, num_workers=num_workers)
        evaluator = Evaluator(returns)
        updater = Updater(approximators, sampler, loss, optimizer, evaluators=[policy_evaluator])

        # initialize RL algorithm
        super(REINFORCE, self).__init__(explorer, evaluator, updater)


# alias
VPG = REINFORCE
