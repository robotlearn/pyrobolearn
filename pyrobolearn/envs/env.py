#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the `Env` class which defines the world, states, and possible rewards. This is the main object
a policy interacts with.

Dependencies:
- `pyrobolearn.worlds`
- `pyrobolearn.states`
- `pyrobolearn.actions`
- (`pyrobolearn.rewards`)
- (`pyrobolearn.envs.terminal_condition`)
"""

import copy
import pickle
import numpy as np
import gym

from pyrobolearn.worlds import World, BasicWorld
from pyrobolearn.states import State
from pyrobolearn.actions import Action
from pyrobolearn.rewards import Reward

from pyrobolearn.terminal_conditions import TerminalCondition
from pyrobolearn.physics import PhysicsRandomizer
from pyrobolearn.states.generators import StateGenerator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Env(gym.Env):  # TODO: make it inheriting the gym.Env
    r"""Environment class.

    This class defines the environment as it described in a reinforcement learning setting [1]. That is, given an
    action :math:`a_t` performed by an agent (i.e. policy), the environment computes and returns the next state
    :math:`s_{t+1}` and reward :math:`r(s_t, a_t, s_{t+1})`. A policy can then interact with this environment,
    and be trained.

    The environment defines the world, the rewards, and the states that are returned by it.
    To allow our framework to be generic and modular, the world, rewards, and states are decoupled from the environment,
    and can be defined outside of this one and then provided as inputs to the `Env` class. That is, we favor
    'composition over inheritance' (see [2]). This is a different approach compared to what is usually done using
    the OpenAI gym framework (see [3]). That said, in order to be compatible with this framework, we inherit from
    the `gym.Env` class (see `core.py` in `https://github.com/openai/gym/blob/master/gym/core.py`).

    References:
        - [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 1998
        - [2] "Wikipedia: Composition over Inheritance", https://en.wikipedia.org/wiki/Composition_over_inheritance
        - [3] "OpenAI gym": https://gym.openai.com/   and    https://github.com/openai/gym
    """

    def __init__(self, world, states, rewards=None, terminal_conditions=None, initial_state_generators=None,
                 physics_randomizers=None, extra_info=None, actions=None):
        """
        Initialize the environment.

        Args:
            world (World): world of the environment. The world contains all the objects (including robots), and has
                access to the simulator.
            states ((list of) State): states that are returned by the environment at each time step.
            rewards (None, Reward): The rewards can be None when for instance we are in an imitation learning setting,
                instead of a reinforcement learning one. If None, only the state is returned by the environment.
            terminal_conditions (None, callable, TerminalCondition, list[TerminalCondition]): A callable function or
                object that check if the policy has failed or succeeded the task.
            initial_state_generators (None, StateGenerator, list[StateGenerator]): state generators which are used
                when resetting the environment to generate the initial states.
            physics_randomizers (None, PhysicsRandomizer, list[PhysicsRandomizer]): physics randomizers. This will be
                called each time you reset the environment.
            extra_info (None, callable): Extra info returned by the environment at each time step.
            actions ((list of) Action): actions that are given to the environment. Note that this is not used here in
                the current environment as it should be the policy that performs the action. This is useful when
                creating policies after the environment (that is, the policy can uses the environment's states and
                actions).
        """
        # Check and set parameters (see corresponding properties)
        self.world = world
        self.states = states
        self.rewards = rewards
        self.terminal_conditions = terminal_conditions
        self.physics_randomizers = physics_randomizers
        self.state_generators = initial_state_generators
        self.extra_info = extra_info if extra_info is not None else lambda: dict()
        self.actions = actions

        # state dictionary which contains at least {'policy': State, 'value': State}
        # if not specified, it will be the same state for the policy and value function approximator
        self._state_dict = None

        # check if we are rendering with the simulator
        self.is_rendering = self.simulator.is_rendering()
        self.rendering_mode = 'human'

        # save the world state in memory
        self.initial_world_state = self.world.save()

    ##############
    # Properties #
    ##############

    @property
    def world(self):
        """Return an instance of the world."""
        return self._world

    @world.setter
    def world(self, world):
        """Set the world."""
        if not isinstance(world, World):
            raise TypeError("Expecting the given 'world' argument to be an instance of `World`, instead got: "
                            "{}".format(type(world)))
        self._world = world
        self.sim = self._world.simulator

    @property
    def simulator(self):
        """Return an instance of the simulator."""
        return self._world.simulator

    @property
    def states(self):
        """Return the states."""
        return self._states

    @states.setter
    def states(self, states):
        """Set the states."""
        if isinstance(states, State):
            states = [states]
        elif isinstance(states, (list, tuple)):
            for idx, state in enumerate(states):
                if not isinstance(state, State):
                    raise TypeError("The {}th item is not an instance of `State`, but instead: "
                                    "{}".format(idx, type(state)))
        else:
            raise TypeError("Expecting the 'states' argument to be an instance of `State` or a list of `State`, "
                            "instead got: {}".format(type(states)))
        self._states = states

    @property
    def state(self):
        """Return the first (combined) state."""
        return self._states[0]

    @property
    def state_dict(self):
        """Return the state dictionary which contains at least the 'policy' and 'value' keys."""
        if self._state_dict is not None:
            return self._state_dict
        states = self.states
        if len(states) == 1:
            states = states[0]
        return {'policy': states, 'value': states}

    @state_dict.setter
    def state_dict(self, state_dict):
        """Set the state dictionary which should contains at least the 'policy' and 'value' keys."""
        if state_dict is not None:
            if not isinstance(state_dict, dict):
                raise TypeError("Expecting the given 'state_dict' to be a dictionary, but got instead: "
                                "{}".format(type(state_dict)))
            for key, value in state_dict.items():
                if isinstance(value, (list, tuple)):
                    for v in value:
                        if not isinstance(v, State):
                            raise TypeError("Expecting the values in the given 'state_dict' to be an instance of "
                                            "`State`, or a list/tuple of them, but got instead: {}".format(type(v)))
                if not isinstance(value, State):
                    raise TypeError("Expecting the value in the given 'state_dict' to be an instance of `State`, or "
                                    "a list/tuple of them, but got instead: {}".format(type(value)))
        self._state_dict = state_dict

    @property
    def state_spaces(self):
        """Return the state space for each state."""
        return [state.merged_space for state in self.states]

    @property
    def state_space(self):
        """Return the state space of the first (combined) state."""
        return self.states[0].merged_space

    # alias
    observations = states
    observation = state
    observation_dict = state_dict
    observation_spaces = state_spaces
    observation_space = state_space

    @property
    def actions(self):
        """Return the actions."""
        return self._actions

    @actions.setter
    def actions(self, actions):
        """Set the actions."""
        if actions is not None:
            # if actions is not a list/tuple, make it a list
            if not isinstance(actions, (list, tuple)):
                actions = [actions]

            # verify the type of each action
            for idx, action in enumerate(actions):
                if not isinstance(action, Action):
                    raise TypeError("The {}th item in 'actions' is not an instance of `Action`, instead got: "
                                    "{}".format(idx, type(action)))

        # set the actions
        self._actions = actions

    @property
    def action(self):
        """Return the first (combined) action."""
        if self.actions is None:
            return None
        return self.actions[0]

    @property
    def action_spaces(self):
        """Return the action space for each action."""
        if self.actions is None:
            return None
        return [action.merged_space for action in self.actions]

    @property
    def action_space(self):
        """Return the action space of the first (combined) action."""
        if self.actions is None:
            return None
        return self.actions[0].merged_space

    @property
    def rewards(self):
        """Return the rewards."""
        return self._rewards

    @rewards.setter
    def rewards(self, rewards):
        """Set the rewards."""
        if rewards is not None:
            if not isinstance(rewards, Reward):
                raise TypeError("Expecting the given 'rewards' argument to be an instance of `Reward` or None, "
                                "instead got: {}".format(type(rewards)))
        self._rewards = rewards

    @property
    def reward_range(self):
        """Return the range of the reward function; a tuple corresponding to the min and max possible rewards"""
        return self._rewards.range

    @property
    def terminal_conditions(self):
        """Return the terminal condition."""
        return self._terminal_conditions

    @terminal_conditions.setter
    def terminal_conditions(self, conditions):
        """Set the terminal condition."""
        if conditions is None:
            conditions = [TerminalCondition()]
        elif isinstance(conditions, TerminalCondition):
            conditions = [conditions]
        elif isinstance(conditions, (list, tuple)):
            for idx, condition in enumerate(conditions):
                if not isinstance(condition, TerminalCondition):
                    raise TypeError("Expecting the {} item in the given terminal conditions to be an instance of "
                                    "`TerminalCondition`, instead got: {}".format(idx, type(condition)))
        else:
            raise TypeError("Expecting the terminal conditions to be an instance of `TerminalCondition`, or a list of "
                            "`TerminalCondition`, but instead got: {}".format(type(conditions)))
        self._terminal_conditions = conditions

    @property
    def physics_randomizers(self):
        """Return the list of physics randomizers used each time we reset the environment."""
        return self._physics_randomizers

    @physics_randomizers.setter
    def physics_randomizers(self, randomizers):
        """Set the physics randomizers."""
        if randomizers is None:
            randomizers = []
        elif isinstance(randomizers, PhysicsRandomizer):
            randomizers = [randomizers]
        elif isinstance(randomizers, (list, tuple)):
            for randomizer in randomizers:
                if not isinstance(randomizer, PhysicsRandomizer):
                    raise TypeError("Expecting the randomizer to be an instance of `PhysicsRandomizer`, instead got "
                                    "{}".format(randomizer))
        else:
            raise TypeError("Expecting the given randomizers to be None, a `PhysicsRandomizer`, or a list of them; "
                            "instead got: {}".format(type(randomizers)))
        self._physics_randomizers = randomizers

    @property
    def state_generators(self):
        """Return the initial state generator instance."""
        return self._state_generators

    @state_generators.setter
    def state_generators(self, generators):
        """Set the initial state generator."""
        if generators is None:
            generators = []
        elif isinstance(generators, StateGenerator):
            generators = [generators]
        elif isinstance(generators, (list, tuple)):
            for generator in generators:
                if not isinstance(generator, StateGenerator):
                    raise TypeError("Expecting the generator to be an instance of `StateGenerator`, instead got "
                                    "{}".format(generator))
        else:
            raise TypeError("Expecting the given generators to be None, a `StateGenerator`, or a list of them; "
                            "instead got: {}".format(type(generators)))
        self._state_generators = generators

    ###########
    # Methods #
    ###########

    @staticmethod
    def _convert_state_to_data(states, convert=True):
        """Convert a `State` to a list of numpy arrays or a numpy array."""
        if convert:
            data = []
            for state in states:
                if isinstance(state, State):
                    state = state.merged_data
                if isinstance(state, list) and len(state) == 1:
                    state = state[0]
                data.append(state)
            if len(data) == 1:
                data = data[0]
            return data
        return states

    def reset(self):
        """
        Reset the environment; reset the world and states.

        Returns:
            list/np.array: list of state data
        """
        # reset world
        self.world.reset()

        # randomize the environment
        for randomizer in self.physics_randomizers:
            randomizer.randomize()

        # generate initial states (states are reset by the states generators)
        for generator in self.state_generators:
            generator()  # reset_state=False)

        # self.world.step()

        # reset states and return first states/observations
        # states = [state.reset(merged_data=True) for state in self.states]
        # print("Reset: ", states)
        states = [state.merged_data for state in self.states]
        return self._convert_state_to_data(states)

    def step(self, actions=None, sleep_dt=None):
        """
        Run one timestep of the environment's dynamics. When end of episode is reached, you are responsible for
        calling `reset()` to reset this environment's state. Accepts an action and returns a tuple (observation,
        reward, done, info).

        Args:
            actions (None, (list of) Action, (list of) np.array): an action provided by the policy(ies) to the
                environment. Note that this is not normally used in this method; calling the actions should be done
                inside the policy(ies), and not in the environment. The policy decides when to execute an action.
                Several problems can appear by providing the actions in the environment instead of letting the policy
                executes them. For instance, think about when there are multiple policies, when using multiprocessing,
                or when the environment runs in real-time. However, if an action is given as a (list of) np.array,
                it will be set as the action data, and the action will be executed. If the action is a (list of) Action,
                it will call each action.
            sleep_dt (float): time to sleep.

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined
              results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # if not isinstance(actions, (list, tuple)):
        #     actions = [actions]
        # if actions is not None and not isinstance(actions, Action):
        #     raise TypeError("Expecting actions to be an instance of Action.")

        # apply each policy's action in the environment
        # for action in actions:
        #    action()
        # TODO: calling the actions should be done inside the policy(ies), and not in the environments. The policy
        #  decides when to execute an action. Think about when there are multiple policies, when using multiprocessing,
        #  or when the environment runs in real-time.
        # if actions is not None and isinstance(actions, Action):
        #     actions()

        # if the actions are provided, set and apply them in the environment
        if actions is not None:
            if isinstance(actions, Action):
                actions()
            elif isinstance(actions, (np.ndarray, int, float, np.integer)) and \
                    isinstance(self.actions, list):  # set the data
                if len(self.actions) == 1:
                    self.actions[0].data = actions
                else:
                    raise ValueError("There are multiple actions defined in the environment, so it is unclear to "
                                     "which action the data should be set to.")
            elif isinstance(actions, (list, tuple)):
                for idx, action in enumerate(actions):
                    if isinstance(action, Action):
                        action()
                    elif isinstance(action, np.ndarray) and self.actions is not None:
                        if len(actions) != len(self.actions):
                            raise ValueError("The number of given actions (={}) is different from the number of "
                                             "actions defined in the environments (={})".format(len(actions),
                                                                                                len(self.actions)))
                        self.actions[idx].data = action
                    else:
                        raise TypeError("Expecting a list of np.array or `Action` instead got: {}".format(type(action)))
            else:
                raise TypeError("Expecting an instance of `Action`, np.array, or a list of the previous ones, but got "
                                "instead: {}".format(type(actions)))

        # perform a step forward in the simulation which computes all the dynamics
        self.world.step(sleep_dt=sleep_dt)

        # compute reward
        # rewards = [reward.compute() for reward in self.rewards]
        rewards = self.rewards() if self.rewards is not None else None

        # compute terminating condition
        done = any([condition() for condition in self.terminal_conditions])

        # get next state/obs for each policy
        # TODO: this should be before computing the rewards as some rewards need the next state
        states = [state(merged_data=True) for state in self.states]
        states = self._convert_state_to_data(states, convert=True)

        # get extra information
        info = self.extra_info()

        return states, rewards, done, info

    def render(self, mode='human'):
        """Renders the environment (show the GUI)."""
        self.is_rendering = True
        self.rendering_mode = mode
        self.sim.render()

    def hide(self):
        """Hide the GUI."""
        self.is_rendering = False
        self.sim.hide()

    def close(self):
        """Close the environment."""
        self.sim.close()

    def seed(self, seed=None):
        """
        Set the given seed for the simulator.

        Args:
            seed (int): seed for the random generator used in the simulator.
        """
        # set the simulator seed
        self.simulator.seed(seed)

        # set the seed for the physics randomizer
        for randomizer in self.physics_randomizers:
            randomizer.seed(seed)

    #############
    # Operators #
    #############

    def __call__(self, actions=None):
        """Alias to `step` method."""
        return self.step(actions)

    def __str__(self):
        """Return a string describing the environment."""
        string = self.__class__.__name__ + '(\n\tworld=' + str(self.world) + ',\n\tstates=' + str(self.states) \
                 + ',\n\trewards=' + str(self.rewards) + '\n)'
        return string

    def __copy__(self):
        """Return a shallow copy of the approximator. This can be overridden in the child class."""
        return self.__class__(world=self.world, states=self.states, rewards=self.rewards,
                              terminal_conditions=self.terminal_conditions,
                              initial_state_generators=self.state_generators,
                              physics_randomizers=self.physics_randomizers,
                              extra_info=self.extra_info, actions=self.actions)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the environment. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        world = copy.deepcopy(self.world, memo)
        states = [copy.deepcopy(state, memo) for state in self.states]
        rewards = None if self.rewards is None else copy.deepcopy(self.rewards, memo)
        terminal_conditions = [copy.deepcopy(condition) for condition in self.terminal_conditions]
        state_generators = [copy.deepcopy(generator, memo) for generator in self.state_generators]
        physics_randomizers = [copy.deepcopy(randomizer, memo) for randomizer in self.physics_randomizers]
        extra_info = copy.deepcopy(self.extra_info)
        actions = None if self.actions is None else [copy.deepcopy(action, memo) for action in self.actions]
        env = self.__class__(world=world, states=states, rewards=rewards, terminal_conditions=terminal_conditions,
                             initial_state_generators=state_generators, physics_randomizers=physics_randomizers,
                             extra_info=extra_info, actions=actions)
        memo[self] = env
        return env


class BasicEnv(Env):
    """Basic Environment class.

    It creates a basic environment with a basic world (a floor and with gravity), no rewards, and no states.
    """

    def __init__(self, sim, states=None, rewards=None, terminal_conditions=None, initial_state_generators=None,
                 physics_randomizers=None, extra_info=None, actions=None):
        world = BasicWorld(sim)
        super(BasicEnv, self).__init__(world, states, rewards, terminal_conditions, initial_state_generators,
                                       physics_randomizers, extra_info, actions)


class GymEnv(gym.Env):
    r"""Gym Environment.

    This is a thin wrapper around a PRL environment to a Gym environment. Notably, we make sure that the action is
    defined in the environment, as in PRL the actions don't have to be specified.

    Few notes with respect to PRL:
    - in PRL Env, you don't have to provide the action space nor the action. The reason is that it is the policy that
      should be aware of the action space.
    - in PRL Env, the returned state data can be a list of state data if the states have different dimensions.
    """

    def __init__(self, prl_env):
        """
        Initialize the Gym PRL Environment.

        Args:
            prl_env (Env): pyrobolearn (PRL) environment.
        """
        # check environment
        if not isinstance(prl_env, Env):
            raise TypeError("Expecting the given 'prl_env' to be an instance of `Env`, instead got: "
                            "{}".format(type(prl_env)))
        self.env = prl_env

        # check that the environment has actions
        if self.env.actions is None:
            raise RuntimeError("Expecting the environment to have actions")

    def __getattr__(self, item):
        """The Gym Env have the same methods and attributes as the PRL Env."""
        return getattr(self.env, item)


# Tests
if __name__ == '__main__':
    from pyrobolearn.simulators import Bullet
    import time

    # create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)
    robot = world.load_robot('coman', fixed_base=True)

    # create states
    states = State()

    # create rewards
    reward = Reward()

    # create Env
    env = Env(world, states, reward)

    # dummy action
    action = Action()

    # run the environment for n steps
    for _ in range(10000):
        env.step(action)
        time.sleep(1./240)
