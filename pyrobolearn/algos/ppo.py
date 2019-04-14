#!/usr/bin/env python
"""Provide the Proximal Policy Optimization algorithm.

The PPO is a model-free, on-policy, gradient-based on-policy algorithm that works with discrete and continuous action
spaces.
"""

from pyrobolearn.algos.rl_algo import GradientRLAlgo, Explorer, Evaluator, Updater

from pyrobolearn.policies import Policy
from pyrobolearn.values import ValueApproximator
from pyrobolearn.actorcritics import ActorCritic

# from pyrobolearn.distributions import GaussianModule, DiagonalCovarianceModule, IdentityModule
from pyrobolearn.exploration import ActionExploration

from pyrobolearn.storages import RolloutStorage
from pyrobolearn.samplers import BatchRandomSampler
from pyrobolearn.estimators import GAE
from pyrobolearn.losses import CLIPLoss, ValueLoss, EntropyLoss
from pyrobolearn.optimizers import Adam

from pyrobolearn import logger


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse", "Ilya Kostrikov", "OpenAI Spinning Up (Josh Achiam)"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PPO(GradientRLAlgo):
    r"""Proximal Policy Optimization

    Type:: model-free, on-policy, gradient-based policy search algorithm for discrete and continuous action spaces.

    This class implements the PPO algorithm which was presented in [1], and is inspired on the implementation of [2, 3].
    Compared to [2, 3], the algorithm is made such that it is more modular and flexible by decoupling and defining the
    various concepts (storages, losses, estimators / returns, policies, value function approximators, and others)
    outside the PPO class, and providing them as input to the constructor and thus privileging composition over
    inheritance.

    Most of the rest of the documentation has been copied-pasted from [3], and is reproduced here for completeness.
    If you use this algorithm, please acknowledge / cite [1, 2, 3].

    Background
    ----------

    "PPO is motivated by the same question as TRPO: how can we take the biggest possible improvement step on a policy
    using the data we currently have, without stepping so far that we accidentally cause performance collapse? Where
    TRPO tries to solve this problem with a complex second-order method, PPO is a family of first-order methods that
    use a few other tricks to keep new policies close to old. PPO methods are significantly simpler to implement, and
    empirically seem to perform at least as well as TRPO.

    There are two primary variants of PPO: PPO-Penalty and PPO-Clip.

    * PPO-Penalty approximately solves a KL-constrained update like TRPO, but penalizes the KL-divergence in the
    objective function instead of making it a hard constraint, and automatically adjusts the penalty coefficient over
    the course of training so that it's scaled appropriately.

    * PPO-Clip doesn't have a KL-divergence term in the objective and doesn't have a constraint at all. Instead
    relies on specialized clipping in the objective function to remove incentives for the new policy to get far
    from the old policy." [3]


    Mathematics
    -----------

    The loss to maximize is given by:

    .. math:: max_{\theta} L(\theta) = max_{\theta} \mathcal{E}[L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 H(\theta)]

    with:
    * Clip loss:
        :math:`L^{CLIP}(\theta) = \mathbb{E}[ \min(r_t(\theta) A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) ]`
    * Value function loss: :math:`L^{VF}(\theta) = (V_{target} - V_\theta(s))^2`
    * Entropy loss: :math:`H(\theta) = - \int p(\theta) \log(p(\theta)) d\theta`
    and where :math:`c_1` and :math:`c_2` are coefficients.

    This algorithm uses an actor-critic model, and a GAE estimator.


    Exploration vs Exploitation
    ---------------------------

    "PPO trains a stochastic policy in an on-policy way. This means that it explores by sampling actions according to
    the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial
    conditions and the training procedure. Over the course of training, the policy typically becomes progressively
    less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the
    policy to get trapped in local optima." [3]


    Properties
    ----------

    pros:
        * this is currently one of the most efficient model-free on-policy search algorithm.
    cons:
        * because it is a gradient-based and an on-policy method, it requires several samples.

    UML: RLAlgo <-- GradientRLAlgo <-- PPO


    Pseudo-algo
    -----------

    Pseudo-algorithm (taken from [3] and reproduce here for completeness)::
        1. Input: initial policy parameters :math:`\theta_0`, initial value function parameters :math:`\phi_0`
        2. for k=0,1,...,num_episodes do
        3.     Exploration: Collect set of trajectories :math:`D_k=\{\tau_i\}` by running policy :math:`\pi_{\theta_k}`
                in the environment.
        4.     Evaluation: Compute rewards-to-go :math:`\hat{R}_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})`,
                evaluate advantage estimates :math:`\hat{A}_t` (using any method of advantage estimation) based on the
                current value function :math:`V_{\phi_k}`.
        5.     Update:
                - Update the policy by maximizing the PPO-Clip objective (using e.g. gradient ascent):
        :math:`\theta_{k+1} = \argmax_\theta \frac{1}{|D_k|T} \sum_{\tau \in D_k} \sum_{t=0}^T
               \min(\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_k}(a_t | s_t)} A^{\pi_{\theta_k}(s_t, a_t), }`
                - Update the value function by regression on the mean-squared error (using e.g. gradient descent):
        :math:`\phi_{k+1} = \argmin_\phi \frac{1}{|D_k|T} \sum_{\tau \in D_k} \sum_{t=0}^T
               (V_{\phi_k}(s_t) - \hat{R}_t)^2`


    References:
        [1] "Proximal Policy Optimization Algorithms", Schulman et al., 2017
        [2] "PyTorch Implementations of Reinforcement Learning Algorithms", Kostrikov, 2018
        [3] OpenAI - Spinning Up: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        [4] "Policy Gradient Algorithms":
            https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

    Implementations:
    - pytorch-a2c-ppo-acktr: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
    - pytorch-rl: https://github.com/khushhallchandra/pytorch-rl
    - DeepRL: https://github.com/ShangtongZhang/DeepRL
    """

    def __init__(self, task, approximators, gamma=0.99, tau=0.95, clip=0.2, lr=5e-4, l2_coeff=0.5, entropy_coeff=0.01,
                 num_workers=1, storage=None):
        """
        Initialize the PPO algorithm.

        Args:
            task (RLTask, Env): RL task/env to run
            approximators (ActorCritic, [Policy, Value]): approximators to optimize
            gamma (float): discount factor (which is a bias-variance tradeoff). This parameter describes how much
                importance has the future rewards we get.
            tau (float): trace-decay parameter (which is a bias-variance tradeoff). If :math:`\tau=1`, this results
                in a Monte Carlo method, while :math:`\tau=0` results in a one-step TD methods.
            clip (float): clip parameter
            lr (float): learning rate
            l2_coeff (float): coefficient for squared-error loss between the target and approximated value functions.
            entropy_coeff (float): coefficient for entropy loss.
            num_workers (int): number of workers (useful when parallelizing the code)
        """
        logger.debug('creating PPO algorithm')

        # create actor critic
        actor_critic = approximators
        if isinstance(approximators, (tuple, list)):
            policy, value = None, None
            for approximator in approximators:
                if isinstance(approximator, Policy):
                    policy = approximator
                elif isinstance(approximator, ValueApproximator):
                    value = approximator
            actor_critic = ActorCritic(policy, value)
        if not isinstance(actor_critic, ActorCritic):
            raise TypeError("Expecting 'actor_critic' to be an instance of ActorCritic")

        # get policy
        policy = actor_critic.actor

        # create exploration strategy (wrap the original policy and specify how to explore)
        # By default, for discrete actions it will use a Categorical distribution and for continuous actions, it will
        # use a Gaussian with a diagonal covariance matrix.
        logger.debug('creating the action exploration strategies for each action')
        exploration = ActionExploration(policy)

        # create storage and estimator
        states, actions = policy.states, policy.actions
        logger.debug('create rollout storage')
        storage = RolloutStorage(num_steps=1000, observation_shapes=states.merged_shape,
                                 action_shapes=actions.merged_shape, num_processes=num_workers)
        logger.debug('create return estimator (GAE)')
        estimator = GAE(storage, gamma=gamma, tau=tau)
        logger.debug('create storage sampler')
        sampler = BatchRandomSampler(storage)

        # create loss
        logger.debug('create loss')
        loss = CLIPLoss(clip=clip) + l2_coeff * ValueLoss() + entropy_coeff * EntropyLoss()

        # create optimizer
        logger.debug('create Adam optimizer')
        optimizer = Adam(learning_rate=lr)

        # define the 3 main steps in RL: explore, evaluate, and update
        logger.debug('create explorer, evaluator, and updater')
        explorer = Explorer(task, exploration, storage, num_workers=num_workers)
        evaluator = Evaluator(estimator)
        updater = Updater(policy, sampler, loss, optimizer)

        # initialize RL algorithm
        super(PPO, self).__init__(explorer, evaluator, updater)
