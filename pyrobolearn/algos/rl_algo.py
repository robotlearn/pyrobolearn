#!/usr/bin/env python
"""Provide the basic components for RL algorithms

Dependencies:
- `pyrobolearn.tasks`
   - `pyrobolearn.approximators` (e.g. `pyrobolean.policies`, `pyrobolearn.values`, `pyrobolearn.models`,...)
   - `pyrobolearn.envs`
"""

import numpy as np
# from pathos.multiprocessing import Pool

from pyrobolearn.algos.explorer import Explorer
from pyrobolearn.algos.evaluator import Evaluator
from pyrobolearn.algos.updater import Updater


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RLAlgo(object):  # Algo):
    r"""Reinforcement Learning Algorithm.

    The RL algorithm takes as input a task (i.e. environment + policy), an optimizer, some hyperparameters (such
    as the number of episodes, rollouts, and timesteps). Optional arguments include a learnable dynamic/transition
    model, a value function approximator, a memory, and an exploration strategy.

    RL problem
    ----------

    The RL problem can be depicted as:

            s_t   \pi(a_t|s_t;\theta)  a_t
               --->  Agent's policy ----
              |                         |
              |                         |
              |                         |
      s_{t+1}, ------  Environment  <---
         r_t       p(s_{t+1}|s_t,a_t)

    where :math:`s_t` is the state/observation, :math:`a_t` is the action, :math:`\pi_{\theta}` is the policy
    parametrized by the parameters :math:`\theta`, :math:`r_t` is the reward returend by the environment,
    and :math:`p(s_{t+1}|s_t,a_t)` is the dynamic model of the environment.

    Note that normally there is a clear distinction between the observation and the state. Also, there is a difference
    between the notation used in optimal control and model-based RL.


    Markov Decision Process (MDP)
    ----------------------------

    A RL problem can be formally formulated as a Markov Decision Process (MDP) which is given as a 6-tuple
    :math:`\{\rho_0, S, A, R, P, \gamma\}`, where :math:`\rho_0` is the initial state distribution, :math:`S` is
    the set of of all valide states, :math:`A` is the set of all valid actions,
    :math:`R: S \times A \times S \rightarrow \mathcal{R}` is the reward function with
    :math:`r_t = R(s_t, a_t, s_{t+1})`, :math:`P: S \times A \rightarrow P(S)` is the state transition probability
    function which describes the dynamic model of the environment, with :math:`p(s_{t+1} | s_t, a_t)` being the
    probability, and :math:`\gamma \in [0,1]` is the discount factor used to calculate the return.


    Goal of RL
    ----------

    The goal of RL is to maximize the expected return given by:

    .. math::

        \max_{\theta} J(\theta) &= \max_{\theta} \int_{\mathcal{T}} p_{\theta}(\tau) R(\tau) d\tau \\
                                &= \mathcal{E}_{\pi_\theta}[]

    where :math:`\mathcal{T}` represents the set of all possible trajectories covered by the policy :math:`\pi_\theta`,
    :math:`p(\tau)` is the probability distribution over the trajectories :math:`\tau = (s_0, a_0, s_1, a_1, ...)`,
    and :math:`R(\tau)` is the total return associated with the trajectory.

    .. math::

        p(\tau) &= p(s_0, a_0, ..., s_{T-1}, a_{T-1}, s_T) \\
                &= p(s_0) \prod_{t=0}^{T-1} p(s_{t+1} | s_t, a_t) \pi_\theta(a_t | s_t)

    where we used the product/chain rule of probability and the Markov property. :math:`p(s_{t+1}|s_t,a_t)`
    represents the dynamics of the environment over which we have no control, and
    :math:`\pi_{\theta}(a_t | s_t) = \pi(a_t | s_t; \theta)` is the policy that is parametrized by :math:`\theta`
    on which we have control over it.


    Taxonomy
    --------

    RL problems can be classified into different categories. We follow the taxonomy described in [3].

    * Model-based vs Model-free
        * Model-based: it knows about the dynamics of the environment (i.e. the transition function :math:`P`), i.e.
            it knows :math:`p(s_{t+1} | s_t, a_t)`. This transition dynamics probability could have been computed
            using mathematical equations or learned from data.
        * Model-free: The dynamic model is not known. Model-free policy search consists of 3 main steps: explore,
            evaluate, and update.
    * Value-based <-- Actor-Critic --> Policy-based
        * Value-based: Value-based means that it determines how good it is to be in a certain state. The policy is
            then inferred from this knowledge.
        * Policy-based: Policy-based (aka Policy search) directly optimizes the agent's policy
            :math:`\pi_\theta(a | s)` which maps the state to action. Policy-based methods can further be subdivided
            into 3 categories: Policy Gradient (PG) vs Expectation-Maximization (EM) vs Information Theory (Inf.Th.)
        * Actor-critic: Actor-Critic combines both previous approaches.
    * On-policy vs Off-policy
        * On-policy: the collected data and the data on which we train the policy is the one collected by the same
            policy.
        * Off-policy: the policy that is being optimized and the policy that explores in the environment are different.
            The former is called the target policy while the latter is the behavior policy which collects the data.
            Off-policy tends to be a little bit slower than on-policy methods.
    * Step-based vs Episode-based
        * Step-based: the exploration is performed in the action space
        * Episode-based: the exploration is performed in the parameter space


    Pseudo-algo
    -----------

    The basic steps of RL algorithms are:
    1. explore in the environment with the current policy and generate samples
    1.5 if model-based, learn a model of the environment
    2. evaluate the policy performance
    3. Update the policy


    Open Problems
    -------------

    - hierarchical
    - exploration
    - algorithms
    - sample efficiency
    - simulation vs reality


    References (tutorials):
        [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 2018
        [2] "Reinforcement Learning", Silver, UC London, 2015
        [3] "A Survey on Policy Search for Robotics", Deisenroth et al., 2013
        [4] "CS294: Deep Reinforcement Learning", Levine et al., UC Berkeley, 2017
        [5] OpenAI - Spinning Up: https://spinningup.openai.com/
    """

    def __init__(self, explorer, evaluator, updater, dynamic_model=None):  # , hyperparameters={}, num_workers=1):
        """
        Initialize the reinforcement learning algorithm.

        Args:
            explorer (Explorer): explorer that specifies how to explore in the environment
            evaluator (Evaluator): evaluate the actions
            updater (Updater): update the approximators (rl, value-functions,...)
            dynamic_model (None): dynamical model
        """

        super(RLAlgo, self).__init__()
        # TODO: think about multiple agents/rewards

        self.explorer = explorer
        self.evaluator = evaluator
        self.updater = updater

        self.env = self.environment
        self.dynamic_model = dynamic_model

        self.best_reward = -np.infty
        self.best_parameters = None

    ##############
    # Properties #
    ##############

    @property
    def explorer(self):
        """Return the explorer instance."""
        return self._explorer

    @explorer.setter
    def explorer(self, explorer):
        """Set the exploration phase."""
        if not isinstance(explorer, Explorer):
            raise TypeError("Expecting the explorer to be an instance of `Explorer`, instead got: "
                            "{}".format(type(explorer)))
        self._explorer = explorer

    @property
    def evaluator(self):
        """Return the evaluator used to evaluate the actions taken by the policy."""
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator):
        """Set the evaluator for the 2nd phase of RL algorithms."""
        if evaluator is not None and not isinstance(evaluator, Evaluator):
            raise TypeError("Expecting the evaluator to be an instance of `Evaluator`, instead got: "
                            "{}".format(type(evaluator)))
        self._evaluator = evaluator

    @property
    def updater(self):
        """Return the updater instance that is used to update the various approximator parameters."""
        return self._updater

    @updater.setter
    def updater(self, updater):
        """Set the updater for the 3rd phase of RL algorithms."""
        if not isinstance(updater, Updater):
            raise TypeError("Expecting the updater to be an instance of `Updater`, instead got: "
                            "{}".format(type(updater)))
        self._updater = updater

    @property
    def task(self):
        """Return the RL task."""
        return self.explorer.task

    @property
    def environment(self):
        return self.task.environment

    @property
    def policy(self):
        """Return the policy."""
        return self.explorer.policy

    @property
    def exploration_strategy(self):
        """Return the exploration strategy."""
        return self.explorer.explorer

    @property
    def estimator(self):
        """Return the estimator."""
        if self.evaluator is not None:
            return self.evaluator.estimator
        return None

    @property
    def storage(self):
        """Return the storage unit."""
        return self.explorer.storage

    @property
    def optimizers(self):
        """Return the optimizers."""
        return self.updater.optimizers

    @property
    def losses(self):
        """Return the losses."""
        return self.updater.losses

    ###########
    # Methods #
    ###########

    def init(self, num_steps, num_rollouts, num_episodes, seed=None, *args, **kwargs):
        """
        Initialize the reinforcement learning algorithm.

        Args:
            num_steps (int): number of step per rollout/trajectory
            num_rollouts (int): number of rollouts/trajectories per episode (default: 1)
            num_episodes (int): number of episodes (default: 1)
            seed (int): random seed
            *args (list): list of optional arguments.
            **kwargs (dict): dictionary of optional arguments.
        """
        pass

    # def init(self, explorer, evaluator, updater):
    #     """Initialize the RL algo."""
    #     self.explorer = explorer
    #     self.evaluator = evaluator
    #     self.updater = updater

    def train(self, num_steps, num_rollouts=1, num_episodes=1, verbose=False, seed=None):
        """
        Train the policy in the provided environment.

        Args:
            num_steps (int): number of step per rollout/trajectory
            num_rollouts (int): number of rollouts/trajectories per episode (default: 1)
            num_episodes (int): number of episodes (default: 1)
            verbose(bool): if True, print details about the optimization process
            seed (int): random seed

        Returns:
            dict: history
        """
        history = {}

        # init algo with the given parameters
        self.init(num_steps=num_steps, num_rollouts=num_rollouts, num_episodes=num_episodes, seed=seed)

        # set the policy in training mode
        self.policy.train()

        # for each episode
        for episode in range(num_episodes):

            # for each rollout
            for rollout in range(num_rollouts):
                # TODO: consider to learn the dynamic model if provided

                # Explore, evaluate, and update
                self.explorer(num_steps, rollout)
                if self.evaluator is not None:
                    self.evaluator()
                loss = self.updater()

                # add the loss in the history
                history.setdefault('loss', []).append(loss)

        # set the policy in test mode
        self.policy.eval()

        return history

    def test(self, num_steps, dt=0., use_terminating_condition=False, render=True):  # , storage):
        """
        Test the policy in the environment; perform one rollout.

        Args:
            num_steps (int): number of steps
            dt (float): time step
            use_terminating_condition (bool): if we should use the terminating condition to end preemptively the
                task if the policy succeeded or failed this last one
            render (bool): render the test phase (default: True)

        Returns:
            list: list of results for each policy at each time step
        """
        results = self.task.run(self, num_steps=num_steps, dt=dt, use_terminating_condition=use_terminating_condition,
                                render=render)
        return results

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a representation string about the object."""
        return self.__class__.__name__

    def __str__(self):
        """Return a string describing the object."""
        return self.__class__.__name__


class GradientRLAlgo(RLAlgo):
    r"""Gradient based reinforcement learning algorithm.

    These methods optimizes directly the policy by computing the gradient of the expected reward.

    .. math:: g = \mathbb{E}[ \sum_{t=0}^\infty \psi_t \nabla_\theta \log \pi_\theta(a_t | s_t) ]

    where:
    * :math:`\pi_\theta(a_t | s_t)` is the policy parametrized by the vector :math:`\theta`. The policy predicts the
        action :math:`a_t` given the state :math:`s_t`.
    * :math:`\psi_t` is a type of return function (such as total reward, value function, advantage function,
        TD residual,...)
    """

    def __init__(self, explorer, evaluator, updater, dynamic_model=None):  # hyperparameters=None)
        super(GradientRLAlgo, self).__init__(explorer, evaluator, updater, dynamic_model)


class EMRLAlgo(RLAlgo):
    r"""Expectation-Maximization reinforcement learning algorithm.
    """

    def __init__(self, task, exploration_strategy, storage, dynamic_model=None):  # hyperparameters=None)
        super(EMRLAlgo, self).__init__(task, exploration_strategy, storage, dynamic_model)
