#!/usr/bin/env python
"""Define the basic Policy class.

A policy couples one or several learning model(s), the state, and action together. In this framework, the policy
usually represents the robot's "brain".

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
- `pyrobolearn.approximators` (and thus `pyrobolearn.models`)
"""

from abc import ABCMeta, abstractmethod
import pickle
import torch

from pyrobolearn.states import State
from pyrobolearn.actions import Action

from pyrobolearn.models import Model
from pyrobolearn.approximators import Approximator, NNApproximator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Policy(object):
    r"""Abstract `Policy` class.

    A policy maps a state to an action, and is often denoted as :math:`\pi_{\theta}(a_t|s_t)`, where :math:`\theta`
    represents the policy parameters. It represents the cognition of the agent(s).
    In our framework, the policy groups the learning model, state, and action objects.

    Specifically, the policy is dissociated from the learning model, as a learning model can be used for different
    purposes. For instance, a neural network can be used to represent a policy but also a value function approximator,
    thus we separate these 2 notions (policy and learning model).

    The policy is also loosely dissociated from the simulator and more specifically from the agent's body, as this last
    one is seen as being part of the environment. The states and actions are what connects the policy with the
    environment (and thus the simulator). The states and actions are given to the policy, and allows to build
    automatically a learning model (if not given) by inferring the dimensions of the inputs and outputs of the model.

    .. note::

        Exploration can be carried out by the policy, by specifying the exploration strategy (that is, exploration
        in the parameter or action space).

    Example::

        # create simulator
        simulator = BulletSim(render=True)

        # create world
        world = BasicWorld(simulator)

        # create robot
        robot = world.loadRobot('robot_name')
        # or load the robot (via urdf) and spawns it in the simulator
        #robot = Robot(simulator)
        #world.loadRobot(robot) # a robot is part of the world (if not done, it will be done inside Env)

        # create states / actions
        states = JntPositionState(robot) + JntVelocityState(robot)
        actions = JntPositionAction(robot)

        # optional: create learning model (if defined, it has to agree with the dimensions of states/actions)
        model = NN(...)

        # create policy (if learning model not defined, it will create it inside)
        policy = Policy(states, actions, model)

        # create rewards/costs (i.e. r(s,a,s')): gives robot, or state/actions
        reward = ForWardProgressReward(robot) - FallenCost(robot) - PowerConsumptionCost(robot)

        # create environment to interact with
        env = Env(world, states, rewards)

        # create and run task
        task = Task(env, policy)
        task.run()

        # Optional: create RL algo (see RL_Algo)

    .. seealso::

        * `state.py`: describes the various states
        * `action.py`: describes the various actions
        * `model.py`: describes the abstract learning model class
        * `exploration.py`: describes how to explore using the policy
    """
    __metaclass__ = ABCMeta

    def __init__(self, states, actions, model=None, rate=1, preprocessors=None, postprocessors=None,
                 distribution=None, *args, **kwargs):
        r"""
        Initialize a policy, the learning model.

        Args:
            states (State): By giving the `states` to the policy, it can automatically infer the type and size/shape of
                       each state, and thus can be used to automatically build a policy. At each step, the `states`
                       are filled by the environment, and read by the policy. The `state` connects the policy with
                       one or several objects (including robots) in the environment.
                       Note that some policies don't use any state information.
            actions (Action): At each step, by calling `policy.act(state)`, the `actions` are computed by the policy,
                        and should be given to the environment. As with the `states`, the type and size/shape of
                        each action can be inferred and could be used to automatically build a policy.
                        The `action` connects the policy with a controllable object (such as a robot) in the
                        environment.
            model (Model, Approximator, None): inner model or approximator
            rate (int): rate at which the policy operates
            distribution:
            args:
            kwargs:
        """
        self.states = states
        self.actions = actions
        self.model = model
        self.train_mode = False
        self.rate = rate
        self.cnt = 0
        self.last_action = None

    ##############
    # Properties #
    ##############

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, states):
        if states is not None:
            if not isinstance(states, State):
                raise TypeError("Expecting states to be an instance of State.")
        self._states = states

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, actions):
        if not isinstance(actions, Action):
            raise TypeError("Expecting actions to be an instance of Action.")
        self._actions = actions

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model is not None and not isinstance(model, Approximator):
            # Try to wrap it with the corresponding Approximator
            if isinstance(model, Model):
                model = Approximator(inputs=self.states, outputs=self.actions, model=model)
            elif isinstance(model, torch.nn.Module):
                model = NNApproximator(inputs=self.states, outputs=self.actions, model=model)
            # else:
            #     raise TypeError("Expecting the model to be an instance of Model.")
        self._model = model

    @property
    def rate(self):
        """Return the rate at which the policy operates."""
        return self._rate

    @rate.setter
    def rate(self, rate):
        """Set the rate at which the policy operates."""
        if not isinstance(rate, int):
            raise TypeError("Expecting the rate to be an integer.")
        self._rate = rate

    @property
    def parameters(self):
        """
        Return an iterator over the learning model parameters.
        """
        if self.model is None:
            return None
        return self.model.parameters()

    @property
    def hyperparameters(self):
        """
        Return an iterator over the learning model hyperparameters.
        """
        if self.model is None:
            return None
        return self.model.hyperparameters()

    @property
    def input_dims(self):
        """
        Return the input dimension of the policy.
        """
        if self.model is None:
            return None
        return self.model.get_input_dims()

    @property
    def output_dims(self):
        """
        Return the output dimension of the policy.
        """
        if self.model is None:
            return None
        return self.model.get_output_dims()

    @property
    def num_parameters(self):
        """Return the total number of parameters"""
        return self.model.num_parameters

    ###########
    # Methods #
    ###########

    def is_deterministic(self):
        """
        Return True if the policy is deterministic; that is, given the same states result in the same actions.
        .. math:: a_t = f(s_t)

        Returns:
            bool: True if the policy is deterministic
        """
        return self.model.is_deterministic()

    def is_stochastic(self):
        """
        Return True if the policy is stochastic; that is, given the same states can result in different actions.
        .. math:: a_t ~ p(a_t|s_t)

        Returns:
            bool: True if the policy is stochastic
        """
        return self.model.is_stochastic()

    def is_parametric(self):
        """
        Return True if the policy is parametric.

        Returns:
            bool: True if the policy is parametric.
        """
        return self.model.is_parametric()

    def is_linear(self):
        """
        Return True if the policy is linear (wrt the parameters). This can be for instance useful for some learning
        algorithms (some only works on linear models).

        Returns:
            bool: True if it is a linear policy
        """
        return self.model.is_linear()

    def is_recurrent(self):
        """
        Return True if the policy is recurrent. This can be for instance useful for some learning algorithms which
        change their behavior when they deal with recurrent learning models.

        Returns:
            bool: True if it is a recurrent policy.
        """
        raise self.model.is_recurrent()

    def get_vectorized_parameters(self, to_numpy=True):
        return self.model.get_vectorized_parameters(to_numpy=to_numpy)

    def set_vectorized_parameters(self, vector):
        self.model.set_vectorized_parameters(vector=vector)

    @abstractmethod
    def act(self, state, deterministic=True, to_numpy=True):
        """
        Perform the action given the state.

        Args:
            state (State): current state
            deterministic (bool): True by default. It can only be set to False, if the policy is stochastic.
            to_numpy (bool): if True, return a np.array

        Returns:
            Action: action
        """
        if self.model is not None:
            if (self.cnt % self.rate) == 0:
                self.last_action = self.model.predict(state, to_numpy=to_numpy)
            self.cnt += 1
            return self.last_action
    # predict = act

    @abstractmethod
    def sample(self, state):
        """
        Given the state, sample from the policy. This only works if the inner model of the policy is stochastic.

        Args:
            state (State, array): current state

        Returns:
            array: sample
        """
        pass

    def train(self, mode=True):
        """
        Set the policy to train mode.

        Args:
            mode (bool): if True, set the policy in train mode.

        Returns:
            None
        """
        self.train_mode = mode

    def reset(self, *args, **kwargs):
        """
        Reset the policy.
        """
        self.model.reset()

    def get_params(self):
        """
        Return the learning model parameters.
        """
        if self.model is None:
            return None
        return self.model.get_params()

    def get_hyperparams(self):
        """
        Return the learning model hyperparameters
        """
        if self.model is None:
            return None
        return self.model.get_hyperparams()

    def save(self, filename):
        """
        Save the policy in the given filename.

        Args:
            filename (str): file to save the policy into

        Returns:
            None
        """
        # self.model.save(filename)
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        """
        Load the policy from the given file.

        Args:
            filename (str): file to load the policy from

        Returns:
            None
        """
        # self.model.load(filename)
        return pickle.load(open(filename, 'rb'))

    #############
    # Operators #
    #############

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    def __str__(self):
        return self.model.__str__()
